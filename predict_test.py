from tqdm import tqdm
import network
import utils
import os
import argparse
import numpy as np
import yaml
import pyvips
import math
import cv2
import pandas as pd


from torch.utils import data
# from datasets import WSIMaskDataset
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

from datasets.wsi_utils import KIDCellDataset

from datasets.wsi import WSIMaskDataset

from torch.utils.data import DataLoader, sampler

from wsi_core import slide_tools
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_dataset import Wsi_Region

from wsi_core.batch_process_utils import initialize_df, load_params, get_simple_loader

from wsi_core.file_utils import save_hdf5

from shapely.geometry import Polygon, MultiPolygon

import kornia

import geojson
from geojson import Feature, FeatureCollection

from preproc_dataset.mat_utils import nuclei_dict_from_mask, nuclei_dict_from_inst

import scipy.io as sio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--dataset", type=str, default='KID-MP-10cell',
                        choices=['KID-MP-10cell'], help='Name of training set')
    parser.add_argument("--config_file", type=str, required=False, default=None,
                        help="path to config file for processing")
    parser.add_argument("--output", type=str, required=False, default=None,
                        help="path to output directory")
    parser.add_argument("--patch_size", type=int, required=False, default=512,
                    help="patch size for segmenting patches")
    parser.add_argument("--batch_size", type=int, required=False, default=64,
                        help="batch size for segmenting patches")
    
    
    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    
    # keep these default values for KID-MP-10cell
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="checkpoint file for inference")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	if args.data_dir is not None:
		config_dict['data_arguments']['data_dir'] = args.data_dir
	if args.ckpt_path is not None:
		config_dict['model_arguments']['ckpt_path'] = args.ckpt_path
	return config_dict


def postprocess_cell_watershed_to_mat(seg_mask, 
                                #   roi, 
                                  max_connected_components=1000):
    # combine all classes into one mask N X C x H x W -> N x 1 x H x W
    # if the mask has at least one non-zero value in C dimension, set it to 1
    # generic cell seg mask
    cell_mask = torch.where(seg_mask > 0., 1., 0.)

    k_3x3 = torch.ones(3,3).to(device)
    k_5x5 = torch.ones(5,5).to(device)

    # remove smaller dots
    opening = kornia.morphology.erosion(cell_mask, kernel=k_3x3)
    opening = kornia.morphology.dilation(opening, kernel=k_3x3)
    opening = kornia.morphology.erosion(opening, kernel=k_3x3)
    opening = kornia.morphology.dilation(opening, kernel=k_3x3)

    sure_bg = kornia.morphology.dilation(opening, kernel=k_3x3)
    # sure_bg = kornia.morphology.dilation(sure_bg, kernel=k_3x3)

    invert_op = torch.ones_like(opening) - opening
    dist_transform = kornia.contrib.distance_transform(invert_op, kernel_size=3)

    # local maxima from distance transform
    dilate_dist = kornia.morphology.dilation(dist_transform, kernel=k_5x5)
    dilate_dist = kornia.morphology.dilation(dist_transform, kernel=k_5x5)
    dist_transform[dilate_dist == 0] = -1
    # these filters were designed from tinkering with the parameters, trial and error.
    # The idea is to find the local maxima and suppress the smaller noisy peaks with morphological operations
    local_max = torch.where(dist_transform >= 0.7*dilate_dist, 1., 0.)
    local_max = kornia.morphology.dilation(local_max, kernel=k_5x5)
    local_max = kornia.morphology.erosion(local_max, kernel=k_5x5)
    local_max = kornia.morphology.erosion(local_max, kernel=k_3x3)
    local_max = kornia.morphology.dilation(local_max, kernel=k_5x5)
    sure_fg = local_max
    
    unknown = sure_bg - sure_fg

    # determining how many connected components there are and labeling them with unique values for each N images
    markers = kornia.contrib.connected_components(sure_fg, num_iterations=150)

    markers_int = torch.zeros_like(markers, dtype=torch.int16)

    for n in range(markers.size(0)):
        # Flatten the tensor for the nth element across C, W, H to find unique values
        unique_values, inverse_indices = torch.unique(markers[n].flatten(), return_inverse=True, sorted=True)

        # if len(unique_values) > max_connected_components, set markers to 0. These are likely to be noise
        if len(unique_values) > max_connected_components:
            markers[n] = 0
            continue
        
        # Map unique_values to their new indices
        markers_int[n] = inverse_indices.view(markers.shape[1], markers.shape[2], markers.shape[3])

    # add 1 to all markers to avoid 0 label
    markers_int += 1

    # mark region of unknown with 0. This is ready for watershedding on the 0 boundary
    markers_int[unknown > 0] = 0

    # process all the tensors to numpy for cv2.watershed on each N (CPU only for cv2, slow)
    np_markers = markers_int.squeeze(1).cpu().numpy().astype(np.int32)
    # distance transform to segment
    np_seg = dist_transform.cpu().numpy().astype(np.uint8)
    # np_seg = cell_mask.cpu().numpy().astype(np.uint8)
    # expand np_seg to 3 channels
    np_seg = np.repeat(np_seg, 3, axis=1)
    # C x W x H -> W x H x C
    np_seg = np_seg.transpose(0, 2, 3, 1)

    watershed_markers_np = np_markers.copy()

    for i in range(np_markers.shape[0]):
        watershed_markers_np[i] = cv2.watershed(np_seg[i], np_markers[i])

    # # relabel the markers based on the majority class of the pixels in the marker on the original segmentation mask
    # watershed_markers_np = watershed_markers.squeeze(1).cpu().numpy()
    watershed_markers = torch.tensor(watershed_markers_np, dtype=torch.int64).unsqueeze(1).to(device)

    # replace border and background with 0
    watershed_markers[watershed_markers == -1] = 1
    watershed_markers[watershed_markers == 1] = 0

    # iterate through batch
    nuclei_data_batch = []
    for i in range(watershed_markers.size(0)):
        inst_label = watershed_markers[i].squeeze(0).cpu().numpy()
        mask = seg_mask[i].squeeze(0).cpu().numpy()
        nuclei_data = nuclei_dict_from_inst(inst_label, mask, metadata=None)
        nuclei_data_batch.append(nuclei_data)

    # ############################################################################################################
    # # partially vectorized code
    # # Find max instance number across watershed markers
    # num_classes = 10
    # max_instance_per_image = torch.amax(watershed_markers, dim=(2, 3), keepdim=True)
    # max_instance_across_all_images = torch.amax(max_instance_per_image)

    # # Initialize class_map
    # class_map = watershed_markers.clone()

    # # Change background (1) to 0 on class_map
    # class_map[class_map == 1] = 0

    # # Create a one-hot encoded version of seg_mask for counting
    # seg_mask_one_hot = torch.nn.functional.one_hot(seg_mask, num_classes=num_classes).permute(0, 1, 4, 2, 3)

    # for i in range(2, max_instance_across_all_images + 1):
    #     # Mask for the current instance
    #     instance_mask = (watershed_markers == i).unsqueeze(2)

    #     # Calculate the sum of classes within the instance mask
    #     class_counts = (seg_mask_one_hot * instance_mask).sum(dim=(3, 4))

    #     # Determine the majority class for each image in the batch
    #     majority_class = class_counts.argmax(dim=2)

    #     # Relabel the instance with the majority class in class_map
    #     for n in range(watershed_markers.size(0)):  # Loop over the batch dimension
    #         class_map[n][watershed_markers[n] == i] = majority_class[n]

    ############################################################################################################

    # # debugging code
    # 1024 == 35
    # 512 == 53, 22, 37
    # j = 39
    # j = 22
    # j = 37
    # j = 53
    # j += 1
    # test_mask = seg_mask.squeeze(1).cpu().numpy()[j]
    # test_roi = roi[j].cpu().numpy().transpose(1, 2, 0) # HWC
    # class_map_fig = class_map.squeeze(1).cpu().numpy()[j]

    # plt.figure(1)
    # plt.imshow(test_mask)
    # plt.figure(2)
    # plt.imshow(test_roi)
    # plt.figure(3)
    # plt.imshow(class_map_fig)

    # # test_mask = cell_mask.squeeze(1).cpu().numpy()[j] # HW
    # t2 = opening.squeeze(1).cpu().numpy()[j] # HW
    # t2 = invert_op.squeeze(1).cpu().numpy()[j] # HW
    # t1 = dist_transform.squeeze(1).cpu().numpy()[j] # HW
    # t2 = dilate_dist.squeeze(1).cpu().numpy()[j] # HW
    # t3 = local_max.squeeze(1).cpu().numpy()[j] # HW
    # # ## test_mask = complement_dist.squeeze(1).cpu().numpy()[j] # HW
    # ## t2 = corners.squeeze(1).cpu().numpy()[j] # HW
    # t2 = sure_fg.squeeze(1).cpu().numpy()[j] # HW
    # t2 = sure_bg.squeeze(1).cpu().numpy()[j] # HW
    # t3 = unknown.squeeze(1).cpu().numpy()[j] # HW
    # t3 = markers_int.squeeze(1).cpu().numpy()[j] # HW
    
    # plt.figure(1)
    # plt.imshow(t1)
    # plt.figure(2)
    # plt.imshow(t2)
    # plt.figure(3)
    # plt.imshow(t3)
    # plt.figure(4)
    # plt.imshow(watershed_markers_np[j])
    # plt.figure(5)
    # plt.imshow(test_roi)

    return nuclei_data_batch


def contour_classified_mask_batch(segmentation_masks, **kwargs):
    N, H, W = segmentation_masks.shape
    contoured_images = np.zeros_like(segmentation_masks, dtype=np.uint8)
    batch_polygons = [None] * N
    for i in range(N):
        contoured_images[i], batch_polygons[i] = contour_classified_mask(segmentation_masks[i], **kwargs)

    return contoured_images, batch_polygons

def contour_classified_mask(segmentation_mask, remove_values=[-1, 0], generate_polygons=True):
    # Get unique classification values excluding the boundary value (-1)
    classification_values = np.unique(segmentation_mask)
    # remove values from the list
    classification_values = [value for value in classification_values if value not in remove_values]
    
    # Create a blank image for drawing contours
    H, W = segmentation_mask.shape
    contoured_image = np.zeros((H, W, 3), dtype=np.uint8)
    image_polygons = []
    
    # Iterate over each classification value
    for value in classification_values:
        # Create a binary mask for the current classification
        binary_mask = (segmentation_mask == value).astype(np.uint8)
        
        # Find contours for the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        if generate_polygons:
            for contour in contours:
                if contour.shape[0] >= 3:  # Contours with less than 3 points cannot form a polygon
                    polygon = Polygon(contour[:, 0, :])
                    image_polygons.append((polygon, value))

        # Draw the contours on the blank image with the classification value
        # need to have less than 254 classes for this to work
        RGB_color = np.array([value, value, value], dtype=np.uint8).tolist()
        # for contour in contours:
        cv2.drawContours(contoured_image, contours, -1, RGB_color, 4)

    
    # select one channel for the grayscale image result
    contoured_image = contoured_image[:,:,0]

    return contoured_image, image_polygons

# for QuPath
def create_geojson_result(polygons_per_image, classif_labels, coord_entries, geojson_output_path=None):
    # assemble the geojson result
    # geojson_dict = {'type': 'FeatureCollection', 'features': []}
    geojson_polys = []
    for i, polygon_list in enumerate(polygons_per_image):
        # for each polygon in image, need to offset coordinates by the patch coordinates in coord_entries
        coord_entry = coord_entries[i]
        x_offset, y_offset = coord_entry[0]['coords']
        for j, (polygon, classif) in enumerate(polygon_list):
            # get the class name
            class_entry = classif_labels[classif]
            if isinstance(class_entry, list):
                class_name = classif_labels[classif][0]
                class_color = classif_labels[classif][1]
            else:
                class_name = class_entry
                # default red color
                class_color = [255, 0, 0]
            # get the contour coordinates
            poly_coords = polygon.exterior.coords
            # offset the coordinates
            coords = [(x + x_offset, y + y_offset) for x, y in poly_coords]
            # create the geojson Feature for QuPath
            feature = Feature(
                            id="PathDetectionObject",
                            geometry=Polygon(coords), 
                            properties={
                                'classification': {"name": class_name, "color": class_color}, 
                                'isLocked': False, 
                                'measurements': []
                                }
                            )
            # feature = {'type': 'Feature', 
                    #    'geometry': {'type': 'Polygon', 'coordinates': [list(coords)]},
                    #    'properties': {'classification': class_name} 
                    #    }
            # geojson_dict['features'].append(feature)
            geojson_polys.append(feature)
    
    if geojson_output_path is not None:
        print('saving geojson to: ', geojson_output_path)
        with open(geojson_output_path, 'w') as f:
            f.write(geojson.dumps(geojson_polys, indent=4))

    return geojson_polys


def collate_mat_masks(batch):
	img = torch.cat([item["image"].unsqueeze(0) for item in batch], dim = 0)
	mat_masks = [item["mask"] for item in batch]
	return [img, mat_masks]

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower().startswith('kid-'):
        classes = KIDCellDataset(opts)
        opts.num_classes = classes.num_classes
        decode_fn = classes.decode_target
        overlay_fn = classes.color_mask_overlay
    else:
        raise ValueError("Unknown dataset: %s" % opts.dataset)
    
    # load the split_df and select the test set column
    split_df = pd.read_csv("E:/Applikate/Kidney-DeepLearning/cell-segmentation/DeepLabv3_results/KID-MP-10cell_512_resnet101/0.7train-0.15val-0.15test_split27.csv")
    test_df = split_df["test_split"]
    # remove nan
    test_df = test_df.dropna()
    pred_dir = "E:/Applikate/Kidney-DeepLearning/mats/512x512_pred"
    os.makedirs(pred_dir, exist_ok=True)

    wsi_dir = "E:\Applikate\Kidney-DeepLearning"
    coord_dir = None
    mask_dir = "E:/Applikate/Kidney-DeepLearning/mats/512x512_stride128"
    process_list = "E:\Applikate\Kidney-DeepLearning\Patch_coords-512\KID-patch_seg-level_process_list.csv"
    # rescale_mpp = True
    desired_mpp = 0.2
    wsi_exten = ['.tif','.svs']
    mask_exten = '.mat'

    test_dst = WSIMaskDataset(opts, wsi_dir, coord_dir, mask_dir, classes=classes.classes, 
                                 process_list = process_list,
                                 wsi_exten=wsi_exten, mask_exten=mask_exten, 
                                 rescale_mpp=True, desired_mpp=desired_mpp, 
                                 is_label=True,
                                 is_label_mat=True,
                                 phase='val',
                                 mask_split_list=test_df.to_list(),
                                 aug=False, 
                                 resolution=512,
                                 one_hot=False,
                                 make_all_pipelines=False)
    
    test_loader = data.DataLoader(
        test_dst, batch_size=64, shuffle=False, 
        num_workers=1, collate_fn=collate_mat_masks)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    
    # preprocessing transform for input patches
    # transform = T.Compose([
    #         T.ToTensor(),
    #     ])
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        raise ValueError("No checkpoint for model found at %s" % opts.ckpt)

    # if opts.save_val_results_to is not None:
    #     os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        for i, seg_data in tqdm(enumerate(test_loader)):
            images, gt_mats = seg_data[0], seg_data[1]
            images = images.to(device)
            batch_mask = model(images).max(1)[1].unsqueeze(1)
            batch_postproc = postprocess_cell_watershed_to_mat(batch_mask, 
                                                        #    images
                                                           )
            for j, (pred_mat, gt_mat) in enumerate(zip(batch_postproc, gt_mats)):
                #  reconstruct name as wsi_name + _coord + x +, + y + _ds + ds
                if pred_mat['inst_map'].max() > 2**16-1:
                    # print('Warning: too many instances in patch, truncating to 65535')
                    raise ValueError('Too many instances in patch')
                pred_mat['inst_map'] = pred_mat['inst_map'].astype(np.uint16)
                pred_mat['class_map'] = pred_mat['class_map'].astype(np.uint8)
                wsi_name = gt_mat['wsiname'][0]
                x = gt_mat['x'][0][0]
                y = gt_mat['y'][0][0]
                ds = gt_mat['ds'][0]
                mat_name = f"{wsi_name}_coord{x},{y}_ds{ds}.mat"
                mat_path = os.path.join(pred_dir, mat_name)
                print(f"Saving {mat_path}")
                sio.savemat(mat_path, pred_mat)




if __name__ == '__main__':
    main()
