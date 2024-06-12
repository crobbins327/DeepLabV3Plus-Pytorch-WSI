from torchvision import transforms
import pandas as pd
import numpy as np
import time
import pdb
import PIL.Image as Image
import h5py
from torch.utils.data import Dataset
import torch
from wsi_core.util_classes import Contour_Checking_fn, isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard
from wsi_core.WholeSlideImage import WholeSlideImage
import pyvips
import os

def default_transforms(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean = mean, std = std)])
    return t

def get_contour_check_fn(contour_fn='four_pt_hard', cont=None, ref_patch_size=None, center_shift=None):
    if contour_fn == 'four_pt_hard':
        cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size, center_shift=center_shift)
    elif contour_fn == 'four_pt_easy':
        cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size, center_shift=0.5)
    elif contour_fn == 'center':
        cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size)
    elif contour_fn == 'basic':
        cont_check_fn = isInContourV1(contour=cont)
    else:
        raise NotImplementedError
    return cont_check_fn


class Wsi_Region(Dataset):
    '''
    args:
        wsi_object: instance of WholeSlideImage wrapper over a WSI
        top_left: tuple of coordinates representing the top left corner of WSI region (Default: None)
        bot_right tuple of coordinates representing the bot right corner of WSI region (Default: None)
        level: downsample level at which to prcess the WSI region
        patch_size: tuple of width, height representing the patch size
        step_size: tuple of w_step, h_step representing the step size
        contour_fn (str): 
            contour checking fn to use
            choice of ['four_pt_hard', 'four_pt_easy', 'center', 'basic'] (Default: 'four_pt_hard')
        transform: custom torchvision transformation to apply 
        custom_downsample (int): additional downscale factor to apply 
        use_center_shift: for 'four_pt_hard' contour check, how far out to shift the 4 points
    '''
    def __init__(self, wsi_object, top_left=None, bot_right=None, level=0, 
                 patch_size = (256, 256), step_size=(256, 256), 
                 contour_fn='four_pt_easy',
                 transform=None, custom_downsample=1, use_center_shift=False, 
                 coords_hdf5_path=None
                 ):
        
        self.custom_downsample = custom_downsample

        # downscale factor in reference to level 0
        self.ref_downsample = wsi_object.level_downsamples[level]
        # patch size in reference to level 0
        self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        
        if self.custom_downsample > 1:
            self.target_patch_size = patch_size
            patch_size = tuple((np.array(patch_size) * np.array(self.ref_downsample) * custom_downsample).astype(int))
            step_size = tuple((np.array(step_size) * custom_downsample).astype(int))
            self.ref_size = patch_size
        else:
            step_size = tuple((np.array(step_size)).astype(int))
            self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        
        self.wsi_path = wsi_object.wsi_path
        self.wsi_reader = wsi_object.wsi_reader

        # slides aren't picklable, so we need to re-open the slide on the first call to __getitem__
        self.wsi = None
        self.current_level = None
        
        self.desired_level = level
        self.patch_size = patch_size


         
        if coords_hdf5_path is not None and os.path.exists(coords_hdf5_path):
            raise NotImplementedError
            # print('loading coordinates from hdf5 file')
            # with h5py.File(coords_hdf5_path, 'r') as f:
            #     coords = f['coords'][:]
        else:
            # generate coordinates if the hdf5 file doesn't exist
            if not use_center_shift:
                center_shift = 0.
            else:
                overlap = 1 - float(step_size[0] / patch_size[0])
                if overlap < 0.25:
                    center_shift = 0.375
                elif overlap >= 0.25 and overlap < 0.75:
                    center_shift = 0.5
                elif overlap >=0.75 and overlap < 0.95:
                    center_shift = 0.5
                else:
                    center_shift = 0.625
                #center_shift = 0.375 # 25% overlap
                #center_shift = 0.625 #50%, 75% overlap
                #center_shift = 1.0 #95% overlap
            
            filtered_coords = []
            filtered_indices = []
            filtered_cont_id = []
            coord_meta_dicts = [] 
            combined_coords = []
            #iterate through tissue contours for valid patch coordinates
            for cont_idx, contour in enumerate(wsi_object.contours_tissue): 
                print('processing {}/{} contours'.format(cont_idx, len(wsi_object.contours_tissue)))
                cont_check_fn = get_contour_check_fn(contour_fn, contour, self.ref_size[0], center_shift)
                coord_results, contour_meta = wsi_object.process_contour(contour, wsi_object.holes_tissue[cont_idx], level, '', cont_idx, 
                                patch_size = patch_size[0], step_size = step_size[0], contour_fn=cont_check_fn,
                                use_padding=False, top_left = top_left, bot_right = bot_right)
                if len(coord_results) > 0:
                    filtered_coords.append(coord_results['coords'])
                    # ij indices relative to the contour bounding box
                    filtered_indices.append(coord_results['grid_ind'])
                    # repeat cont_idx for each coordinate
                    filtered_cont_id.extend([cont_idx] * len(coord_results['coords']))
                    coord_meta_dicts.append(contour_meta)
            
            coords=np.vstack(filtered_coords)
            grid_indices = np.vstack(filtered_indices)
            combined_coords = [{"coords": coords[i], "grid_ind": grid_indices[i], "cont_idx": filtered_cont_id[i]} for i in range(len(coords))]

        self.coord_entries = combined_coords
        self.coords_meta_attr = WholeSlideImage.reorg_coord_attr(coord_meta_dicts)
        print('filtered a total of {} coordinates'.format(len(self.coord_entries)))
        
        # apply transformation
        if transform is None:
            self.transforms = default_transforms()
        else:
            self.transforms = transform

    @staticmethod
    def reorgainize_coord_entries(coord_entries):
        # coord entries are collated into an array or list of dictionaries
        # array([[{'coords': array([10626,  2493]), 'grid_ind': array([0, 3]), 'cont_idx': 0}],
        #         [{'coords': array([10626,  3005]), 'grid_ind': array([0, 4]), 'cont_idx': 0}],
        #         ...
        #         ])
        # N x 1 array
        # this function reorganizes the entries into several dictionaries for each field that can be written to an hdf5 dataset
        keys = coord_entries[0][0].keys()
        coord_entries_dict = {key: [] for key in keys}
        for entry in coord_entries:
            for key in keys:
                coord_entries_dict[key].append(entry[0][key])
        # vstack the lists into arrays for hdf5
        for key in keys:
            coord_entries_dict[key] = np.vstack(coord_entries_dict[key])

        return coord_entries_dict
        

    #TODO: padding if image request is larger than the WSI bounds
    def read_region(self, x, y, w, h, level=0):
        if self.current_level != level:
            self.wsi = self.wsi_reader.slide2vips(level=level)
            self.current_level = level
        # use crop to get region for vips image
        # try:
        return self.wsi.crop(x, y, w, h)
        # except Exception as e:
        #     print('Error reading region: ', e)
        #     print('location: ', x, y, w, h, level)
        #     return None
        # could alternatively read region/fetch for vips image and store them
    
    def __len__(self):
        return len(self.coord_entries)
    
    def __getitem__(self, idx):
        if self.wsi is None:
            self.wsi = self.wsi_reader.slide2vips(level=self.desired_level)
            self.current_level = self.desired_level
        coord_entry = self.coord_entries[idx]
        coord = coord_entry["coords"]
        read_reg = self.read_region(coord[0], coord[1], self.patch_size[0], self.patch_size[0], level=self.desired_level)
        patch = np.array(read_reg)
        if self.custom_downsample > 1:
            patch = patch.resize(self.target_patch_size)
        patch = self.transforms(patch).unsqueeze(0)
        return patch, coord_entry