import os
import argparse
import numpy as np
import yaml
import math
import cv2

from jaya_kidney_seg.unet import UNet


from torch.utils import data

# from datasets import WSIMaskDataset
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ARTERIES_MODEL_PATH = "jaya_kidney_seg/models/he-normal-arteries-10X_unet_best_model.pth"
DIST_TUB_MODEL_PATH = "jaya_kidney_seg/models/he-normal-distal-tubules-10X_unet_best_model.pth"
PROX_TUB_MODEL_PATH = "jaya_kidney_seg/models/he-normal-proximal-tubules-10X_unet_best_model.pth"
GLOM_CAPSULE_MODEL_PATH = "jaya_kidney_seg/models/he-normal-glom-capsule-5X_unet_best_model.pth"


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True, help="path to a single image or image directory")
    parser.add_argument(
        "--config_file", type=str, required=False, default=None, help="path to config file for processing"
    )
    parser.add_argument("--output", type=str, required=False, default=None, help="path to output directory")
    parser.add_argument("--patch_size", type=int, required=False, default=256, help="patch size for segmenting patches")
    parser.add_argument("--batch_size", type=int, required=False, default=64, help="batch size for segmenting patches")

    # jayapandian kidney histo seg models

    # parser.add_argument(
    #     "--model", type=str, default="deeplabv3plus_resnet101", choices=available_models, help="model name"
    # )
    # models are hardcoded for now

    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    return parser


def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict["exp_arguments"]["save_exp_code"] = args.save_exp_code
    if args.overlap is not None:
        config_dict["patching_arguments"]["overlap"] = args.overlap
    if args.data_dir is not None:
        config_dict["data_arguments"]["data_dir"] = args.data_dir
    if args.ckpt_path is not None:
        config_dict["model_arguments"]["ckpt_path"] = args.ckpt_path
    return config_dict


def postprocess_cell_watershed_v2(
    seg_mask,
    #   roi,
    max_connected_components=1000,
):
    # combine all classes into one mask N X C x H x W -> N x 1 x H x W
    # if the mask has at least one non-zero value in C dimension, set it to 1
    # generic cell seg mask
    cell_mask = torch.where(seg_mask > 0.0, 1.0, 0.0)

    k_3x3 = torch.ones(3, 3).to(device)
    k_5x5 = torch.ones(5, 5).to(device)

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
    local_max = torch.where(dist_transform >= 0.7 * dilate_dist, 1.0, 0.0)
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

    ############################################################################################################
    # partially vectorized code
    # Find max instance number across watershed markers
    num_classes = 10
    max_instance_per_image = torch.amax(watershed_markers, dim=(2, 3), keepdim=True)
    max_instance_across_all_images = torch.amax(max_instance_per_image)

    # Initialize class_map
    class_map = watershed_markers.clone()

    # Change background (1) to 0 on class_map
    class_map[class_map == 1] = 0

    # Create a one-hot encoded version of seg_mask for counting
    seg_mask_one_hot = torch.nn.functional.one_hot(seg_mask, num_classes=num_classes).permute(0, 1, 4, 2, 3)

    for i in range(2, max_instance_across_all_images + 1):
        # Mask for the current instance
        instance_mask = (watershed_markers == i).unsqueeze(2)

        # Calculate the sum of classes within the instance mask
        class_counts = (seg_mask_one_hot * instance_mask).sum(dim=(3, 4))

        # Determine the majority class for each image in the batch
        majority_class = class_counts.argmax(dim=2)

        # Relabel the instance with the majority class in class_map
        for n in range(watershed_markers.size(0)):  # Loop over the batch dimension
            class_map[n][watershed_markers[n] == i] = majority_class[n]

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

    return class_map


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
    contoured_image = contoured_image[:, :, 0]

    return contoured_image, image_polygons


def segment_from_patches(wsi_object, model, batch_size, mask_save_path, decode_fn, overlay_fn, **wsi_kwargs):
    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    reorg_coord_entries_fn = Wsi_Region.reorgainize_coord_entries
    coords_meta_attr = roi_dataset.coords_meta_attr
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=4)
    print("total number of patches to process: ", len(roi_dataset))
    num_batches = len(roi_loader)
    print("number of batches: ", len(roi_loader))
    init_h5 = True
    wsi_polygons_per_patch = []
    all_coord_entries = []
    with torch.no_grad():
        model = model.eval()
        for idx, (rois, coord_entries) in enumerate(roi_loader):
            rois = rois.to(device)
            batch_mask = model(rois).max(1)[1].unsqueeze(1)  # N x CHW

            # # debugging code
            # 1024 == 35
            # 512 == 53, 22, 37
            j = 0
            test_mask = batch_mask.squeeze(1).cpu().numpy()[j]
            test_roi = rois[j].cpu().numpy().transpose(1, 2, 0)  # HWC

            plt.figure(1)
            plt.imshow(test_mask)
            plt.figure(2)
            plt.imshow(test_roi)
            j += 1

            batch_contour_np, contour_list = contour_classified_mask_batch(batch_postproc.squeeze(1).cpu().numpy())
            # batch_contour_color_np = decode_fn(batch_contour_np)
            # batch_mask_color_np = decode_fn(batch_postproc.squeeze(1).cpu().numpy())
            batch_roi_np = (255 * rois.cpu().numpy().transpose(0, 2, 3, 1)).astype("uint8")
            batch_overlay_np = overlay_fn(batch_roi_np, batch_contour_np, a=1.0)

            if idx % math.ceil(num_batches * 0.05) == 0:
                print("processed {} / {}".format(idx, num_batches))

            # Saving image patch outputs for creating image based overlay
            if mask_save_path is not None:
                coord_entries_dict = reorg_coord_entries_fn(coord_entries)

                # asset_dict = {'mask_gray': batch_contour_np,
                #             'mask_color': batch_contour_color_np,
                #             'overlay':  batch_overlay_np
                #             }
                asset_dict = {"overlay": batch_overlay_np}
                # append coord_entries_dict to asset_dict
                asset_dict.update(coord_entries_dict)

                if init_h5:
                    attr_dict = {"coords": coords_meta_attr}
                    save_hdf5(mask_save_path, asset_dict, attr_dict, mode="w")
                    init_h5 = False
                else:
                    save_hdf5(mask_save_path, asset_dict, mode="a")

            # assemble the polygon list of coordinates across batches
            wsi_polygons_per_patch.extend(contour_list)
            all_coord_entries.extend(coord_entries)

    return wsi_polygons_per_patch, all_coord_entries


# for QuPath
def create_geojson_result(polygons_per_image, classif_labels, coord_entries, geojson_output_path=None):
    # assemble the geojson result
    # geojson_dict = {'type': 'FeatureCollection', 'features': []}
    geojson_polys = []
    for i, polygon_list in enumerate(polygons_per_image):
        # for each polygon in image, need to offset coordinates by the patch coordinates in coord_entries
        coord_entry = coord_entries[i]
        x_offset, y_offset = coord_entry[0]["coords"]
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
                    "classification": {"name": class_name, "color": class_color},
                    "isLocked": False,
                    "measurements": [],
                },
            )
            # feature = {'type': 'Feature',
            #    'geometry': {'type': 'Polygon', 'coordinates': [list(coords)]},
            #    'properties': {'classification': class_name}
            #    }
            # geojson_dict['features'].append(feature)
            geojson_polys.append(feature)

    if geojson_output_path is not None:
        print("saving geojson to: ", geojson_output_path)
        with open(geojson_output_path, "w") as f:
            f.write(geojson.dumps(geojson_polys, indent=4))

    return geojson_polys


def main():
    opts = get_argparser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    if opts.config_file and os.path.isfile(opts.config_file):
        config_path = os.path.join(opts.config_file)
        config_dict = yaml.safe_load(open(config_path, "r"))
        config_dict = parse_config_dict(opts, config_dict)

        for key, value in config_dict.items():
            if isinstance(value, dict):
                print("\n" + key)
                for value_key, value_value in value.items():
                    print(value_key + " : " + str(value_value))
            else:
                print("\n" + key + " : " + str(value))

        args = config_dict
        patch_args = argparse.Namespace(**args["patching_arguments"])
        # data_args = argparse.Namespace(**args['data_arguments'])
        model_args = args["model_arguments"]
        model_args.update({"n_classes": args["exp_arguments"]["n_classes"]})
        model_args = argparse.Namespace(**model_args)
        # exp_args = argparse.Namespace(**args['exp_arguments'])
        # heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
        # sample_args = argparse.Namespace(**args['sample_arguments'])
    else:
        #  use default values
        patch_args = argparse.Namespace(patch_size=opts.patch_size, overlap=0, patch_level=0, custom_downsample=4.0)

    # preprocessing transform for input patches
    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print(
        "patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}".format(
            patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]
        )
    )
    blocky_wsi_kwargs = {
        "top_left": None,
        "bot_right": None,
        "patch_size": patch_size,
        "step_size": patch_size,
        "custom_downsample": patch_args.custom_downsample,
        "level": patch_args.patch_level,
        "use_center_shift": False,
        "transform": transform,
    }

    def_seg_params = {
        "seg_level": -1,
        "sthresh": 15,
        "mthresh": 11,
        "close": 2,
        "use_otsu": False,
        "keep_ids": "none",
        "exclude_ids": "none",
        "ref_patch_size": patch_args.patch_size,
    }
    def_filter_params = {"a_t": 50.0, "a_h": 8.0, "max_n_holes": 10}
    def_vis_params = {"vis_level": -1, "line_thickness": 250}
    def_patch_params = {"use_padding": True, "contour_fn": "four_pt"}

    if isinstance(opts.input, list):
        slides = []
        for data_dir in opts.input:
            slides.extend(os.listdir(data_dir))
    else:
        if os.path.isdir(opts.input):
            slides = sorted(os.listdir(opts.input))
        else:
            slides = [os.basename(opts.input)]

    if opts.output is not None:
        root_output_dir = opts.output
    else:
        if os.path.isdir(opts.input):
            root_output_dir = os.path.join(opts.input, "output_jaya")
        else:
            root_output_dir = os.path.join(os.path.dirname(opts.input), "output_jaya")

    os.makedirs(root_output_dir, exist_ok=True)

    slide_ext = [".svs", ".tif", ".tiff"]
    # filter out non-slide files by extension
    slides = [slide for slide in slides if slide.endswith(tuple(slide_ext))]
    df = initialize_df(
        slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False
    )

    mask = df["process"] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print("\nlist of slides to process: ")
    print(process_stack.head(len(process_stack)))

    # Set up model (all models... probably a better way to do this instead of evaluating all at once)
    # dist_tub_ckpt = torch.load(DIST_TUB_MODEL_PATH, map_location=lambda storage, loc: storage)
    # dist_tub_model = UNet(
    #     n_classes=dist_tub_ckpt["n_classes"],
    #     in_channels=dist_tub_ckpt["in_channels"],
    #     padding=dist_tub_ckpt["padding"],
    #     depth=dist_tub_ckpt["depth"],
    #     wf=dist_tub_ckpt["wf"],
    #     up_mode=dist_tub_ckpt["up_mode"],
    #     batch_norm=dist_tub_ckpt["batch_norm"],
    # ).to(device)
    # dist_tub_model.load_state_dict(dist_tub_ckpt["model_dict"])
    # dist_tub_model.eval()

    # prox_tub_ckpt = torch.load(PROX_TUB_MODEL_PATH, map_location=lambda storage, loc: storage)
    # prox_tub_model = UNet(
    #     n_classes=prox_tub_ckpt["n_classes"],
    #     in_channels=prox_tub_ckpt["in_channels"],
    #     padding=prox_tub_ckpt["padding"],
    #     depth=prox_tub_ckpt["depth"],
    #     wf=prox_tub_ckpt["wf"],
    #     up_mode=prox_tub_ckpt["up_mode"],
    #     batch_norm=prox_tub_ckpt["batch_norm"],
    # ).to(device)
    # prox_tub_model.load_state_dict(prox_tub_ckpt["model_dict"])
    # prox_tub_model.eval()

    glom_ckpt = torch.load(GLOM_CAPSULE_MODEL_PATH, map_location=lambda storage, loc: storage)
    glom_model = UNet(
        n_classes=glom_ckpt["n_classes"],
        in_channels=glom_ckpt["in_channels"],
        padding=glom_ckpt["padding"],
        depth=glom_ckpt["depth"],
        wf=glom_ckpt["wf"],
        up_mode=glom_ckpt["up_mode"],
        batch_norm=glom_ckpt["batch_norm"],
    ).to(device)
    glom_model.load_state_dict(glom_ckpt["model_dict"])
    glom_model.eval()

    # arteries_ckpt = torch.load(ARTERIES_MODEL_PATH, map_location=lambda storage, loc: storage)
    # arteries_model = UNet(
    #     n_classes=arteries_ckpt["n_classes"],
    #     in_channels=arteries_ckpt["in_channels"],
    #     padding=arteries_ckpt["padding"],
    #     depth=arteries_ckpt["depth"],
    #     wf=arteries_ckpt["wf"],
    #     up_mode=arteries_ckpt["up_mode"],
    #     batch_norm=arteries_ckpt["batch_norm"],
    # ).to(device)
    # arteries_model.load_state_dict(arteries_ckpt["model_dict"])
    # arteries_model.eval()

    for i in range(len(process_stack)):
        wsi_path = os.path.join(opts.input, process_stack.loc[i, "slide_id"])
        slide_file = os.path.basename(wsi_path)
        print("\nprocessing: ", slide_file)

        slide_id = slide_tools.get_name(slide_file)
        # TODO:
        separate_outputs = False
        if separate_outputs:
            output_dir = os.path.join(root_output_dir, slide_id)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = root_output_dir

        mask_file = os.path.join(output_dir, slide_id + "_mask.pkl")

        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params["keep_ids"])
        if len(keep_ids) > 0 and keep_ids != "none":
            seg_params["keep_ids"] = np.array(keep_ids.split(",")).astype(int)
        else:
            seg_params["keep_ids"] = []

        exclude_ids = str(seg_params["exclude_ids"])
        if len(exclude_ids) > 0 and exclude_ids != "none":
            seg_params["exclude_ids"] = np.array(exclude_ids.split(",")).astype(int)
        else:
            seg_params["exclude_ids"] = []

        for key, val in seg_params.items():
            print("{}: {}".format(key, val))

        for key, val in filter_params.items():
            print("{}: {}".format(key, val))

        for key, val in vis_params.items():
            print("{}: {}".format(key, val))

        print("Initializing WSI object")
        wsi_object = WholeSlideImage(wsi_path)
        if seg_params["seg_level"] < 0:
            best_level = wsi_object.find_downsample_level(32)
            seg_params["seg_level"] = best_level

        if os.path.exists(mask_file):
            print("loading existing mask file: ", mask_file)
            wsi_object.initSegmentation(mask_file)
        else:
            result = wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
            if result is False:
                print("No tissue found in slide, skipping processing...")
                continue
            wsi_object.saveSegmentation(mask_file)
        print("Done!")

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        mask_path = os.path.join(output_dir, "{}_mask.jpg".format(slide_id))
        if vis_params["vis_level"] < 0:
            best_level = wsi_object.find_downsample_level(32)
            vis_params["vis_level"] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)

        output_h5_path = os.path.join(output_dir, slide_id + ".h5")
        # output_h5_path = None

        ##### check if h5_features_file exists ######
        # if not os.path.isfile(output_h5_path) :
        polygons_per_patch, coord_entries = segment_from_patches(
            wsi_object=wsi_object,
            model=glom_model,
            batch_size=opts.batch_size,
            mask_save_path=output_h5_path,
            decode_fn=None,
            overlay_fn=None,
            **blocky_wsi_kwargs,
        )

        # save overlay image jpg
        overlay_img = wsi_object.visSegmentationMask(
            mask_hdf5_path=output_h5_path,
            vis_level=0,
            mask_id="overlay",
            blank_canvas=False,
            background_color=(255, 255, 255),
        )
        # save with jpg compression set to 95
        overlay_dir = os.path.join(output_dir, "overlay_only")
        os.makedirs(overlay_dir, exist_ok=True)
        overlay_img.save(os.path.join(overlay_dir, "{}_cell_overlay.jpg".format(slide_id)), quality=95)

        # TODO: remove h5 file after processing
        remove_h5 = True
        if remove_h5:
            os.remove(output_h5_path)

        # additional post processing of polygons to merge overlapping polygons based on coord entries
        # wsi_polygons_per_patch = merge_overlapping_polygons(wsi_polygons_per_patch, all_coord_entries)

        # assemble GeoJSON from list of polygons and classify based on the classification value/dataset
        geojson_output_path = os.path.join(output_dir, slide_id + "_cell_geojson.json")
        geojson_polys = create_geojson_result(
            polygons_per_patch, classes.classif_labels, coord_entries, geojson_output_path=geojson_output_path
        )

        # read the h5 file and visualize mask output using method within the WholeSlideImage class
        # seg_img = wsi_object.visSegmentationMask(mask_hdf5_path=output_h5_path, vis_level=0, mask_id="overlay", blank_canvas=False, background_color=(255,255,255))
        # seg_img.save(os.path.join(output_dir, '{}_mask.jpg'.format(slide_id)))


if __name__ == "__main__":
    main()
