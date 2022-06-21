from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
from collections import namedtuple
import torch
import cv2
import albumentations as A
import pandas as pd
import random
import h5py
import openslide
import pyvips
import math
from torch.nn.functional import one_hot as oneHot

#----------------------------------------------------------------------------
class WSIMaskDataset(Dataset):
    def __init__(self, 
                 args,
                 wsi_dir,                   # Path to WSI directory.
                 coord_dir,                 # Path to h5 coord database.
                 mask_dir,
                 classes,
                 process_list = None,       #Dataframe path of WSIs to process and their seg_levels/downsample levels that correspond to the coords
                 wsi_exten = '.svs',
                 mask_exten = '.png',
                 max_coord_per_wsi = 'inf',
                 rescale_mpp = False,
                 desired_mpp = 0.25,
                 # random_seed = 0,
                 load_mode = 'openslide',
                 make_all_pipelines = False,
                 unlabel_transform=None, 
                 # latent_dir=None, 
                 is_label=True, 
                 phase='train',
                 mask_split_list = None,
                 aug=False, 
                 resolution=1024,
                 one_hot = True
                 ):

        self.args = args
        self.is_label = is_label
        
        #Grayscale value of masks and corresponding classes, could load from json file
        # KIDCellClass = namedtuple('KIDCellClass', ['name', 'mask_value', 'id', 'train_id', 'category', 'category_id',
        #                                                  'has_instances', 'ignore_in_eval', 'color'])
        # #ignore index is 255
        # classes = [
        #     KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
        #     KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
        #     KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 255, 0)),
        #     KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, False, (255, 153, 102)),
        #     KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (255, 0, 255)),
        #     KIDCellClass('DCT Nuclei',     98,  5,  3, 'Nuclei', 1, True, False, (0, 0, 128)),
        #     KIDCellClass('Endothelial',   117,  6,  4, 'Nuclei', 1, True, False, (0, 128, 128)),
        #     KIDCellClass('Fibroblast',    137,  7,  5, 'Nuclei', 1, True, False, (235, 206, 155)),
        #     KIDCellClass('Mesangial',     156,  8,  6, 'Nuclei', 1, True, False, (255, 255, 0)),    
        #     KIDCellClass('Parietal cells',176,  9,  7, 'Nuclei', 1, True, False, (58, 208, 67)),    
        #     KIDCellClass('Podocytes',     196, 10, 8, 'Nuclei', 1, True, False, (0, 255, 255)),  
        #     KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (179, 26, 26)),   
        #     KIDCellClass('Tubule Nuclei', 235, 12, 9, 'Nuclei', 1, True, False, (130, 91, 37)),    
        #     ]
        
        self.classes = classes
        self.one_hot = one_hot
        
        #For preparing the masks to seg ID images
        train_ids, index = np.unique(np.array([c.train_id for c in classes]), return_index=True)
        colors = [c.color for c in classes]
        self.train_id_to_color = [colors[index[i]] for i,t in enumerate(train_ids) if (t != -1 and t != 255)]
        #Color for ignore IDs
        self.train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = np.array(self.train_id_to_color)
        
        self.id_to_train_id = np.array(([c.train_id for c in classes]))
        self.val_to_id_dict = dict([(c.mask_value, c.id) for c in classes])
        
        self.num_classes = len(train_ids[train_ids!=255])
        
        # try:
            # random.seed(random_seed)
        # except Exception as e:
            # print(e)
            # random.seed(0)
        self.wsi_dir = wsi_dir
        self.wsi_exten = wsi_exten
        self.mask_exten = mask_exten
        self.coord_dir = coord_dir
        self.max_coord_per_wsi = max_coord_per_wsi
        if process_list is None:
            self.process_list = None
        else:
            self.process_list = pd.read_csv(process_list)
        self.patch_size = resolution
        self.rescale_mpp = rescale_mpp
        self.desired_mpp = desired_mpp
        self.load_mode = load_mode
        self.make_all_pipelines = make_all_pipelines
        #Implement labels here..
        #Need to load the wsi_pipelines after init for multiprocessing?
        self.wsi_pipelines = None

        if is_label == True:
            # self.latent_dir = latent_dir
            self.mask_split_list = mask_split_list
            assert isinstance(mask_dir, str)
            if os.path.isdir(mask_dir):
                self.mask_dir = mask_dir
            else:
                raise ValueError('{} does not exist. Verify mask_dir...'.format(mask_dir))
            #Load the coordinate dic & wsi dicts for the labeled images only...
            #Need a function that looks at masks in mask_dir, pulls out wsi_names and coords from filename
            self.coord_dict, self.wsi_names, self.wsi_props = self.createLabeledWSIData()
            
        else:
            self.coord_dict, self.wsi_names, self.wsi_props = self.createWSIData()
        
        self.data_size = len(self.coord_dict)
        print('Number of WSIs:', len(self.wsi_names))
        print('Number of patches:', self.data_size)
        if self.is_label:
            i_img, i_mask = self._load_raw_image(0, load_one=True)
            raw_shape = [self.data_size] + list(i_img.shape) 
        else:
            raw_shape = [self.data_size] + list(self._load_raw_image(0, load_one=True).shape)
        print('Raw shape of dataset:', raw_shape)
        if resolution is not None and (raw_shape[1] != resolution or raw_shape[2] != resolution):
            raise IOError('Image files do not match the specified resolution')
        #Trying to resolve picking of this dictionary for multiprocessing.....
        #Maybe there's a better way... maybe just load one image or add a 'test' parameter?
        del self.wsi_pipelines
        self.wsi_pipelines = None

        #__get_item__ params
        self.phase = phase
        self.aug = aug
        if aug == True:
            self.aug_t = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            #More conservative aug of rotating ever 90 degrees
                            A.OneOf([
                                A.RandomRotate90(p=1),
                                A.Sequential([A.RandomRotate90(p=1),
                                              A.RandomRotate90(p=1)], p=1),
                                A.Sequential([A.RandomRotate90(p=1),
                                              A.RandomRotate90(p=1),
                                              A.RandomRotate90(p=1)], p=1),
                                ], p=0.75),
                            #The image is rotated in random angles 0 to 360deg. 
                            #May work fine because a lot of whitespace actually exists in core biopsies.
                            #However, I do not want the img generator to produce these images, just want the segmentation branch to learn from them
                            A.ShiftScaleRotate(shift_limit=0.125,
                                                scale_limit=0.2 * math.log(2),
                                                rotate_limit=0,
                                                border_mode=cv2.BORDER_REFLECT_101,
                                                p=0.6),
                            A.ShiftScaleRotate(shift_limit=0,
                                               scale_limit=0,
                                               rotate_limit=360,
                                               border_mode=cv2.BORDER_CONSTANT,
                                               value=[255,255,255],
                                               mask_value=0,
                                               p=0.75),
                            A.ColorJitter(brightness=0.3,
                                          contrast=0.5 * math.log(2),
                                          saturation=0.18,
                                          hue=0.18,
                                          p=0.25)
                            
                    ])

        self.unlabel_transform = unlabel_transform
    
    def createLabeledWSIData(self):
        #Really only care about the files in the mask_dir
        mask_files = sorted([x for x in os.listdir(self.mask_dir) if x.endswith(self.mask_exten)])
        #Could also thin out by process list....
        
        if self.mask_split_list is not None:
            #Only load mask_files intersecting mask_split_list
            mask_files = list(set(mask_files).intersection(self.mask_split_list))
            
            
        #Will use all WSIs that have labels, regardless of process list.... simpler but needs to be modified to use only the process list files to control the number of labels used during training
        wsi_names_noext = sorted(list(set([m.split('_coord')[0] for m in mask_files])))
        #Get the real wsi_names from wsi_dir using a lookup. This is needed for getting correct extensions...
        wsi_names = sorted([w for w in os.listdir(self.wsi_dir) if os.path.splitext(w)[0] in wsi_names_noext])
        temp_wsi_dict = dict(zip(wsi_names_noext,wsi_names))
        #Extract wsi_names and coords from filename
        coords = np.array([list(map(float,m.split('_coord')[1].split('_')[0].split(','))) for m in mask_files])
        downsamples = np.array([float(m.split('_ds')[1].split(self.mask_exten)[0]) for m in mask_files])
        #Duplicate downsample columns
        downsamples = np.transpose([downsamples] * 2)
        #Adjust coord by mask downsample --> ASSUMES THAT YOU MUST RESCALE THE MPP IF MASKS CONTAIN DOWNSAMPLES != 1
        coords = (coords * downsamples).astype(int)
        #Combine coords, mask_file name, and wsi_name into tuple
        tups = [(coords[i], m, temp_wsi_dict[m.split('_coord')[0]]) for i,m in enumerate(mask_files)]
        #very scary one liner to split mask filename to get list of coords to int, mask_name, and wsi_name
        # tups = [(list(map(int,m.split('_')[1].split(self.mask_exten)[0].split(','))), m, temp_wsi_dict[m.split('_')[0]]) for m in mask_files]
        #Get the desired seg level for the patching based on process list
        wsi_props = {}
        for wsi_name in wsi_names:
            mpp = None
            seg_level = 0
            if self.process_list is not None:
                seg_level = int(self.process_list.loc[self.process_list['slide_id']==wsi_name,'seg_level'].iloc[0])
                if self.rescale_mpp and 'MPP' in self.process_list.columns:
                    mpp = float(self.process_list.loc[self.process_list['slide_id']==wsi_name,'MPP'].iloc[0])
                    seg_level = 0
                #if seg_level != 0:
                #    print('{} for {}'.format(seg_level, wsi_name))
            if self.rescale_mpp and mpp is None:
                try:
                    wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
                    mpp = float(wsi.properties['openslide.mpp-x'])
                    seg_level = 0
                except Exception as e:
                    print(e)
                    print(wsi_name)
                    raise ValueError('Cannot find slide MPP from process list ["MPP"] or Openslide properties. Set rescale_mpp to False to avoid this error or add slide MPPs to process list')
            wsi_props[wsi_name] = (seg_level, mpp)
            
        coord_dict = {}
        for i,t in enumerate(tups):
            #Make key a string so that it is less likely to have hash collisions...
            coord_dict[str(i)] = t
        
        return coord_dict, wsi_names, wsi_props
        
        
    def createWSIData(self):
        if self.process_list is None:
            #Only use WSI that have coord files....
            all_coord_files = sorted([x for x in os.listdir(self.coord_dir) if x.endswith('.h5')])
        else:
            #Only use WSI that coord files aren't excluded and are in coord_dir
            wsi_plist = list(self.process_list.loc[~self.process_list['exclude_ids'].isin(['y','yes','Y']),'slide_id'])
            coord_plist = sorted([os.path.splitext(x)[0]+'.h5' for x in wsi_plist])
            all_coord_files = sorted([x for x in os.listdir(self.coord_dir) if x.endswith('.h5') and x in coord_plist])
        #Get WSI filenames from path that have coord files/in process list
        wsi_names = sorted([w for w in os.listdir(self.wsi_dir) if w.endswith(tuple(self.wsi_exten)) and os.path.splitext(w)[0]+'.h5' in all_coord_files])
            
        #Get corresponding coord h5 files using WSI paths
        h5_names = [os.path.splitext(wsi_name)[0]+'.h5' for wsi_name in wsi_names]
        #Loop through coord files, get coord length, randomly choose X coords for each wsi (max_coord_per_wsi)
        coord_dict = {}
        wsi_props = {}
        # wsi_number = 0
        for h5, wsi_name in zip(h5_names, wsi_names):
            #All h5 paths must exist....
            h5_path = os.path.join(self.coord_dir, h5)
            with h5py.File(h5_path, "r") as f:
                attrs = dict(f['coords'].attrs)
                seg_level = int(attrs['patch_level'])
                dims = attrs['level_dim']
                #patch_size = attrs['patch_size']
                dset = f['coords']
                max_len = len(dset)
                if max_len < float(self.max_coord_per_wsi):
                    #Return all coords
                    coords = dset[:]
                else:
                    #Randomly select X coords
                    rand_ind = np.sort(random.sample(range(max_len), int(self.max_coord_per_wsi)))
                    coords = dset[rand_ind]

            #Get the desired seg level for the patching based on process list
            mpp = None
            seg_level = 0
            if self.process_list is not None:
                seg_level = int(self.process_list.loc[self.process_list['slide_id']==wsi_name,'seg_level'].iloc[0])
                if self.rescale_mpp and 'MPP' in self.process_list.columns:
                    mpp = float(self.process_list.loc[self.process_list['slide_id']==wsi_name,'MPP'].iloc[0])
                    seg_level = 0
                #if seg_level != 0:
                #    print('{} for {}'.format(seg_level, wsi_name))
            if self.rescale_mpp and mpp is None:
                try:
                    wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
                    mpp = float(wsi.properties['openslide.mpp-x'])
                    seg_level = 0
                except Exception as e:
                    print(e)
                    print(wsi_name)
                    raise ValueError('Cannot find slide MPP from process list ["MPP"] or Openslide properties. Set rescale_mpp to False to avoid this error or add slide MPPs to process list')
            
            #Check that coordinates and patch resolution is within the dimensions of the WSI... slow but only done once at beginning
            del_index = []
            # print(wsi_name)
            for i,coord in enumerate(coords):
                #Check that coordinates are inside dims
                changed = False
            #   old_coord = coord.copy()
                if coord[0]+self.patch_size > dims[0]:
                    coord[0] = dims[0]-self.patch_size
                #   print('X not in bounds, adjusting')
                    changed = True
                if coord[1]+self.patch_size > dims[1]:
                    coord[1] = dims[1]-self.patch_size
                #   print('Y not in bounds, adjusting')
                    changed = True
                if changed:
                #   print("Changing coord {} to {}".format(old_coord, coord))
                    coords[i] = coord
            
            if len(del_index) > 0:
                print('Removing {} coords that have black or white patches....'.format(len(del_index)))
                coords = np.delete(coords, del_index, axis=0)    
            
            #Store as dictionary with tuples {0: (coord, wsi_number), 1: (coord, wsi_number), etc.}
            dict_len = len(coord_dict)
            for i in range(coords.shape[0]):
                #Make key a string so that it is less likely to have hash collisions...
                coord_dict[str(i+dict_len)] = (coords[i], wsi_name)
            wsi_props[wsi_name] = (seg_level, mpp)
            #Storing number/index because smaller size than string??
            # wsi_number += 1
            
        return coord_dict, wsi_names, wsi_props
    
    #Making a one-hot encoded matrix of mask labels
    # def _mask_labels(self, mask_np):
    #     label_size = len(self.color_map.keys())
    #     class_keys = list(self.class_val.keys())
    #     labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
    #     if self.class_group == None:
    #         for i in range(label_size):
    #             labels[i][mask_np==self.class_val[class_keys[i]]] = 1.0
    #     else:
    #         for i in range(label_size):
    #             group = self.class_group[i]
    #             # print("group", group)
    #             for j in group:
    #                 # print("adding", class_keys[j])
    #                 labels[i][mask_np==self.class_val[class_keys[j]]] = 1.0
            
    #     return labels
    
    def makeOneHot(self, mask_seg):    
        # targets_extend=targets.clone()
        # print(targets_extend.get_device())
        # mask_seg.unsqueeze_(1) # convert to 1x1xHxW
        # one_hot_mask = torch.zeros((1, self.num_classes, mask_seg.size(1), mask_seg.size(2)), dtype=mask_seg.dtype, device=mask_seg.device)
        # one_hot_mask.scatter_(1, mask_seg.unsqueeze(1), 1)  #returns 1xCxHxW, but expecting CxHxW
        # one_hot_mask.squeeze_(0)   #CxHxW
        
        # Alternative
        one_hot_mask = oneHot(mask_seg, num_classes=self.num_classes)
        one_hot_mask.squeeze_(0) #HxWxC
        one_hot_mask = one_hot_mask.permute(2,0,1) #CxHxW
        return one_hot_mask
    
    #For converting the class value colors into integer values (0,64,128,192,255 -> 0,1,2,3,4)
    #This will be converted to a one-hot encoding matrix later for ADA
    def _mask_seg(self, mask_np):
        val_keys = list(self.val_to_id_dict.keys())
        mask_seg = np.zeros((1, mask_np.shape[0], mask_np.shape[1]))
        for i in range(len(val_keys)):
            mask_seg[0][mask_np==val_keys[i]] = self.val_to_id_dict[val_keys[i]]
        #Then convert IDs to train IDs
        mask_seg = mask_seg.astype(int)
        mask_seg = self.id_to_train_id[mask_seg]
        return mask_seg
    
    
    def decode_target(self, mask_seg):
        """decode semantic mask to RGB image"""
        #Set ignore ids to ignore color at end of array
        mask_seg[mask_seg == 255] = len(self.train_id_to_color)-1
        
        return self.train_id_to_color[mask_seg]
    
    #img must be np.uint8 at this point, mask_seg is the argmax version
    #can be CxHxW or NxCxHxW depending on img and mask_seg dimensions
    def color_mask_overlay(self, img, mask_seg, a=0.5):
        if type(img) == torch.Tensor:
            masked = img.detach().clone()
            for key in range(self.num_classes):
                if key != 0:
                    masked[mask_seg==key] = torch.tensor(self.train_id_to_color[key], dtype=torch.float)
            color_overlay = masked*a + img*(1-a)
            return color_overlay.numpy().astype(np.uint8)
        elif type(img) == np.ndarray:
            masked = np.copy(img)
            for key in range(self.num_classes):
                if key != 0:
                    masked[mask_seg==key] = self.train_id_to_color[key]
            color_overlay = masked*a + img*(1-a)
            return np.array(color_overlay).astype(np.uint8)
        else:
            raise ValueError('img of type {} not handled in color_mask_overlay()...'.format(type(img)))
        
        
        
    @staticmethod
    def preprocess(img):
        image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5), inplace=True)     #Normalize between -1 and 1
                        # transforms.Normalize(mean=(0.485, 0.456, 0.406),
                        #                      std=(0.229, 0.224, 0.225), 
                        #                      inplace=True)
                        #Using real data to normalize dist...
                        # transforms.Normalize((0.8153510093688965,
                        #                       0.6476525664329529,
                        #                       0.7707882523536682), 
                        #                       (0.035145699977874756,
                        #                       0.05645135045051575,
                        #                       0.028033018112182617), 
                        #                       inplace=True)
                    ]
                )
        img_tensor = image_transform(img)
        # normalize
        # img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        # img_tensor = (img_tensor - 0.5) / 0.5

        return img_tensor
    
    @staticmethod    
    def adjPatchOOB(wsi_dim, coord, patch_size):
        #wsi_dim = (wsi_width, wsi_height)
        #coord = (x, y) with y axis inverted or point (0,0) starting in top left of image
        #patchsize = integer for square patch only
        #assume coord starts at (0,0) in line with original WSI,
        #therefore the patch is only out-of-bounds if the coord+patchsize exceeds the WSI dimensions
        #check dimensions, adjust coordinate if out of bounds
        coord = [int(coord[0]), int(coord[1])] 
        if coord[0]+patch_size > wsi_dim[0]:
            coord[0] = int(wsi_dim[0] - patch_size)
        
        if coord[1]+patch_size > wsi_dim[1]:
            coord[1] = int(wsi_dim[1] - patch_size) 
        
        return tuple(coord)

    def scalePatch(self, wsi, dims, coord, input_mpp=0.5, desired_mpp=0.25, patch_size=512, eps=0.05, level=0):
        desired_mpp = float(desired_mpp)
        input_mpp = float(input_mpp)
        #downsample > 1, upsample < 1
        factor = desired_mpp/input_mpp
        #Openslide get dimensions of full WSI
        # dims = wsi.level_dimensions[0]
        if input_mpp > desired_mpp + eps or input_mpp < desired_mpp - eps:
            #print('scale by {:.2f} factor'.format(factor))
            # if factor > 1
            #input mpp must be smaller and therefore at higher magnification (e.g. desired 40x vs input 60x) and vice versa
            #approach: shrink a larger patch by factor to the desired patch size or enlarge a smaller patch to desired patch size
            #if factor > 1 and you are downsampling the image, it can be faster to load the downsample level that is closest to the factor
            #get the level that is closest to the factor... really only care if factor > 2 because tiled images increment downsample levels in factors of 2 or 4 typically.
            downsample_at_new_level = 1
            if factor >= 2 and self.load_mode == 'openslide':
                # print('Factor:', factor)
                #Downsamples aren't integers in TCGA data..... but typically they are in increments of 2 anyways.
                level = wsi.get_best_level_for_downsample(int(math.ceil(factor))+0.5)
                downsample_at_new_level = wsi.level_downsamples[level]
                #update factor to scale based on new level. If factor was 5 and the downsample at new level is 4, 
                #then you need to still scale scaled_psize/(downsample*patch_size) == (5*1024)/(4*1024) = 1.25
                factor = factor/downsample_at_new_level
                # print('Adj Factor:', factor)
                # print('Level:', level)
                # print('Downsample at new level:', downsample_at_new_level)
            #Don't know how I could do this in pyvips unless I can get the downsample level metadata...... could try and check if the pyvips image was openslide compatible...?
            if factor >= 1.25 or factor <= 0.75:
                scaled_psize = int(patch_size*factor)
                #check and adjust dimensions of coord based on scaled patchsize relative to level 0
                coord = self.adjPatchOOB(dims, coord, int(scaled_psize*downsample_at_new_level))
                adj_patch = self._load_patch(wsi, level, coord, scaled_psize, dims=dims)
                #shrink patch down to desired mpp if factor > 1
                #enlarge if factor < 1
                #Could implement fully in vips...
                patch = cv2.resize(adj_patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                return patch
            else:
                coord = self.adjPatchOOB(dims, coord, patch_size)
                patch = self._load_patch(wsi, level, coord, patch_size, dims=dims)
                return patch    
        else: 
            #print('skip scaling factor {:.2f}. input um per pixel ({}) within +/- {} of desired MPP ({}).'.format(factor, input_mpp, eps, desired_mpp))
            coord = self.adjPatchOOB(dims, coord, patch_size)
            patch = self._load_patch(wsi, level, coord, patch_size, dims=dims)
            return patch
    
    def _load_wsi_pipelines(self, load_wsi_by_name = None):
        #Create all the image pipelines in a dictionary
        wsi_pipelines = {}
        
        if load_wsi_by_name is not None:
            if isinstance(load_wsi_by_name, list):
                load_WSIs = load_wsi_by_name
            else:
                load_WSIs = [load_wsi_by_name]
        else:
            load_WSIs = self.wsi_names
        
        if self.load_mode == 'openslide':
            for wsi_name in load_WSIs:
                seg_level, mpp = self.wsi_props[wsi_name]
                wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
                dims = wsi.level_dimensions[seg_level]
                wsi_pipelines[wsi_name] = wsi, dims
            return wsi_pipelines
        else:
            raise ValueError('Load mode {} not supported! Modes {} are supported.'.format(self.load_mode, ['openslide']))

    def _load_one_wsi(self, wsi_name):        
        if self.load_mode == 'openslide':
            seg_level, mpp = self.wsi_props[wsi_name]
            wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
            dims = wsi.level_dimensions[seg_level]
            return wsi, dims
        else:
            raise ValueError('Load mode {} not supported! Modes {} are supported.'.format(self.load_mode, ['openslide']))
    
    def _load_patch(self, wsi, level, coord, patch_size, dims=None):
        if self.load_mode == 'openslide':
            patch = np.array(wsi.read_region(coord, level, (patch_size, patch_size)).convert('RGB'))
        else:
            raise ValueError('Load mode {} not supported! Modes {} are supported.'.format(self.load_mode, ['openslide']))
        return patch
    
    def _load_raw_image(self, raw_idx, load_one=False):
        
        if self.is_label:
            coord, mask_name, wsi_name = self.coord_dict[str(raw_idx % self.data_size)]
        else:
            coord, wsi_name = self.coord_dict[str(raw_idx % self.data_size)]
        
        seg_level, mpp = self.wsi_props[wsi_name]
        
        #Load wsi first...
        if self.make_all_pipelines:
            if self.wsi_pipelines is None:
                #load pipelines first
                if load_one:
                    #For the test image for init
                    self.wsi_pipelines = self._load_wsi_pipelines(load_wsi_by_name=wsi_name)
                else:
                    self.wsi_pipelines = self._load_wsi_pipelines()
            wsi, dims = self.wsi_pipelines[wsi_name]
        else:
            wsi, dims = self._load_one_wsi(wsi_name)
        
        #Load wsi patch
        if self.rescale_mpp:
            if mpp is None and self.load_mode == 'openslide':
                try:
                    mpp = wsi.properties['openslide.mpp-x']
                except Exception as e:
                    print(e)
                    print(wsi_name)
                    raise ValueError('Cannot find slide MPP from process list or Openslide properties. Set rescale_mpp to False to avoid this error or add slide MPPs to process list.')
            elif mpp is None and self.load_mode != 'openslide':
                raise ValueError('Cannot find slide MPP from process list. Cannot use mode load_mode {} if MPP not in process list. Change load_mode to "openslide", set rescale_mpp to False, or slide MPPs to process list to avoid error.'.format(self.load_mode))
            img = self.scalePatch(wsi=wsi, dims=dims, coord=coord, input_mpp=mpp, desired_mpp=self.desired_mpp, patch_size=self.patch_size, level=seg_level) 
        else:
            img = self._load_patch(wsi, seg_level, coord, self.patch_size, dims=dims)
        
        if self.is_label:
            #Load label mask
            mask = np.array(Image.open(os.path.join(self.mask_dir, mask_name)).convert('L'))
            return img, mask
        else:
            return img

    def __len__(self):
        if hasattr(self.args, 'n_gpu') == False:
            return self.data_size
        # make sure dataloader size is larger than batchxngpu size
        return max(self.args.batch*self.args.n_gpu, self.data_size)
    
    def __getitem__(self, idx):
        if self.is_label:
            img, mask = self._load_raw_image(idx, load_one=False)
            if (self.phase == 'train' or self.phase == 'train-val') and self.aug:
                augmented = self.aug_t(image=img, mask=mask)
                aug_img_pil = Image.fromarray(augmented['image'])
                # apply pixel-wise transformation
                img_tensor = self.preprocess(aug_img_pil)

                mask_np = np.array(augmented['mask'])
                
                if self.one_hot:
                    mask_seg = torch.from_numpy(self._mask_seg(mask_np))
                    labels = self.makeOneHot(mask_seg)
                    mask_tensor = labels.to(dtype=torch.float32)
                    # mask_tensor = torch.tensor(labels, dtype=torch.float32)
                    #All the 0 become -1 .... why? better gradients for GANs because the -1 and 1 work better than 0 and 1?
                    # mask_tensor = (mask_tensor - 0.5) / 0.5
                else:
                    mask_seg = self._mask_seg(mask_np)
                    #use sparse categorical loss, less information content
                    mask_tensor = torch.tensor(mask_seg, dtype=torch.float32)

            else:
                #When phase is val, just preprocess image & mask normally
                img_pil = Image.fromarray(img)
                img_tensor = self.preprocess(img_pil)
                if self.one_hot:
                    mask_seg = torch.from_numpy(self._mask_seg(mask_np))
                    labels = self.makeOneHot(mask_seg)
                    mask_tensor = labels.to(dtype=torch.float32)
                    # mask_tensor = torch.tensor(labels, dtype=torch.float32)                    #All the 0 become -1 .... why? better gradients for GANs because the -1 and 1 work better than 0 and 1?
                    # mask_tensor = (mask_tensor - 0.5) / 0.5
                else:
                    mask_seg = self._mask_seg(mask)
                    #use sparse categorical loss, less information content
                    mask_tensor = torch.tensor(mask_seg, dtype=torch.float32)
            
            return {
                'image': img_tensor,
                'mask': mask_tensor
            }
                
                
        else:
            img = self._load_raw_image(idx, load_one=False)
            img_pil = Image.fromarray(img)
            if self.unlabel_transform is None:
                img_tensor = self.preprocess(img_pil)
            else:
                img_tensor = self.unlabel_transform(img_pil)
            return {
                'image': img_tensor,
            }
        
    
        
if __name__ == '__main__':
    import argparse
    import os
    import shlex
    import matplotlib.pyplot as plt
    # from torchvision import utils
    from collections import OrderedDict
    os.chdir('/home/cjr66/project/DeepLabV3Plus-Pytorch-WSI')
    import utils.utils
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.n_gpu = 4
    args.batch = 16
    wsi_dir = '/home/cjr66/project/KID-DeepLearning/KID-Images-pyramid'
    coord_dir = '/home/cjr66/project/KID-DeepLearning/Patch_coords-1024/MP_KPMP_all-patches-stride256'
    mask_dir = '/home/cjr66/project/KID-DeepLearning/Labeled_patches/MP_256x256_stride64'
    process_list = '/home/cjr66/project/KID-DeepLearning/proc_info/MP_only-KID_process_list.csv'
    rescale_mpp = True
    desired_mpp = 0.2
    wsi_exten = ['.tif','.svs']
    
    KIDCellClass = namedtuple('KIDCellClass', ['name', 'mask_value', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    #ignore index is 255
    classes = [
        KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
        KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
        KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 255, 0)),
        KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, False, (255, 153, 102)),
        KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (255, 0, 255)),
        KIDCellClass('DCT Nuclei',     98,  5,  3, 'Nuclei', 1, True, False, (0, 0, 128)),
        KIDCellClass('Endothelial',   117,  6,  4, 'Nuclei', 1, True, False, (0, 128, 128)),
        KIDCellClass('Fibroblast',    137,  7,  5, 'Nuclei', 1, True, False, (235, 206, 155)),
        KIDCellClass('Mesangial',     156,  8,  6, 'Nuclei', 1, True, False, (255, 255, 0)),    
        KIDCellClass('Parietal cells',176,  9,  7, 'Nuclei', 1, True, False, (58, 208, 67)),    
        KIDCellClass('Podocytes',     196, 10, 8, 'Nuclei', 1, True, False, (0, 255, 255)),  
        KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (179, 26, 26)),   
        KIDCellClass('Tubule Nuclei', 235, 12, 9, 'Nuclei', 1, True, False, (130, 91, 37)),    
        ]
    
    
    from sklearn.model_selection import train_test_split
    
    mask_files = sorted([x for x in os.listdir(mask_dir) if x.endswith('.png')])
    #0.7, 0.15, 0.15 splits
    y = 0.15
    seed=27
    m_train, m_test = train_test_split(mask_files, test_size=y, random_state=seed)
    m_train, m_val = train_test_split(m_train, test_size=y/(1-y), random_state=seed)
    
    kidData = WSIMaskDataset(args, wsi_dir, coord_dir, mask_dir, classes=classes, 
                             process_list = process_list,
                             wsi_exten=wsi_exten, rescale_mpp=True, desired_mpp=desired_mpp, 
                             is_label=True,
                             phase='train',
                             mask_split_list= m_train,
                             aug=True,
                             resolution=256,
                             one_hot=False)
    
    # denorm = utils.Denormalize(mean=[0.8153510093688965,
    #                                   0.6476525664329529,
    #                                   0.7707882523536682], 
    #                             std=[0.035145699977874756,
    #                                 0.05645135045051575,
    #                                 0.028033018112182617])
    # denorm = utils.Denormalize(mean=[0.5,
    #                                   0.5,
    #                                   0.5], 
    #                             std=[0.5,
    #                                 0.5,
    #                                 0.5])
    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
    #                            std=[0.229, 0.224, 0.225])
    
    import network
    
    train_ids, index = np.unique(np.array([c.train_id for c in classes]), return_index=True)
    
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    
    parser = argparse.ArgumentParser()
    opts = parser.parse_args()
    opts.model = 'deeplabv3plus_resnet101'
    opts.num_classes = len(train_ids[train_ids!=255])
    opts.output_stride = 16
    opts.separable_conv = True
    
    
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, pretrained_backbone=True)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    
    imgs = []
    labs = []
    for i in range(40,60,1):
        dget = kidData.__getitem__(i)
        img, mask = dget['image'], dget['mask']
        imgs.append(img.unsqueeze(0))
        labs.append(mask.unsqueeze(0))
    
    #Test if loss function is compatible
    in_imgs = torch.cat(imgs,dim=0)
    gt_labs = torch.cat(labs, dim=0)
    if gt_labs.dim() == 4:
        gt_labs = gt_labs.squeeze(1).to(dtype=torch.long)
    model.train()
    preds = model(in_imgs)
    loss = criterion(preds, gt_labs)
    color_masks = kidData.decode_target(gt_labs.to(torch.int))
    color_mask_overlay = kidData.color_mask_overlay(in_imgs.permute(0,2,3,1)*255, gt_labs.to(torch.int), a=0.5)
    in_imgs = in_imgs.permute(0,2,3,1)
    for i in range(len(in_imgs)):
        
        # imag_orig = (denorm(img.numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
        # imag_orig = (img.numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
        # imag = img.numpy().transpose(1, 2, 0)
        # # arg_mask = torch.argmax(mask, dim=0)
        # color_mask = kidData.decode_target(mask)
        # color_mask_overlay = kidData.color_mask_overlay(imag_orig, arg_mask, a = 0.7).astype(np.uint8)
        plt.figure()
        plt.subplot(1,4,1)
        plt.imshow(in_imgs[i])
        plt.subplot(1,4,2)
        plt.imshow((in_imgs[i]*255).numpy().astype(np.uint8))
        plt.subplot(1,4,3)
        plt.imshow(color_masks[i])
        plt.subplot(1,4,4)
        # plt.imshow(imag_orig)
        # plt.imshow(color_mask, alpha=0.6)
        plt.imshow(color_mask_overlay[i])
    