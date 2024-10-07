import numpy as np
import cv2
import math
import time
import os
import multiprocessing as mp
from PIL import Image
import h5py


from wsi_core import slide_io, slide_tools
from wsi_core.file_utils import load_pkl, save_pkl, update_attr_hdf5, save_hdf5
from wsi_core.wsi_utils import (
    savePatchIter_bag_hdf5,
    initialize_hdf5_bag,
    coord_generator,
    isBlackPatch,
    isWhitePatch,
    screen_coords,
    to_percentiles,
)
from wsi_core.util_classes import (
    isInContourV1,
    isInContourV2,
    isInContourV3_Easy,
    isInContourV3_Hard,
    Contour_Checking_fn,
    isInContourV3_Grid,
    skip_contour_check,
)
import matplotlib.pyplot as plt

# class for handling whole slide images using vips


class WholeSlideImage(object):

    def __init__(self, path, **kwargs):
        """
        Args:
            path (str): fullpath to WSI file
        """

        self.wsi_path = path
        self.name = slide_tools.get_name(path)
        slide_reader_cls = slide_io.get_slide_reader(path)
        self.wsi_reader = slide_reader_cls(path)

        self.wsi = self.wsi_reader.slide2vips(level=0)
        self.current_level = 0
        self.level_dim = self.wsi_reader.metadata.slide_dimensions
        self.level_downsamples = self.wsi_reader.metadata.slide_downsamples

        self.contours_tissue = None
        self.contours_tumor = None
        self.hdf5_file = None

    # TODO: implement padding if read region is out of bounds
    def read_region(self, x, y, w, h, level=0):
        if self.current_level != level:
            self.wsi = self.wsi_reader.slide2vips(level=level)
            self.current_level = level
        # use crop to get region for vips image
        return self.wsi.crop(x, y, w, h)
        # could alternatively read region/fetch for vips image and store them

    def find_downsample_level(self, target_downsample):
        downsample_diffs = np.abs(np.array(self.level_downsamples) - target_downsample)
        return np.argmin(downsample_diffs)

    def initSegmentation(self, mask_file):
        # load segmentation results from pickle file
        import pickle

        asset_dict = load_pkl(mask_file)
        self.holes_tissue = asset_dict["holes"]
        self.contours_tissue = asset_dict["tissue"]

    def saveSegmentation(self, mask_file):
        # save segmentation results using pickle
        asset_dict = {"holes": self.holes_tissue, "tissue": self.contours_tissue}
        save_pkl(mask_file, asset_dict)

    def segmentTissue(
        self,
        seg_level=0,
        sthresh=20,
        sthresh_up=255,
        mthresh=7,
        close=0,
        use_otsu=False,
        filter_params={"a_t": 100, "a_h": 16, "max_n_holes": 20},
        ref_patch_size=512,
        exclude_ids=[],
        keep_ids=[],
    ):
        """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """

        def _filter_contours(contours, hierarchy, filter_params):
            """
            Filter contours by: area.
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
            all_holes = []

            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                # actual contour
                cont = contours[cont_idx]
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # take contour area (includes holes)
                a = cv2.contourArea(cont)
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                if a == 0:
                    continue
                if tuple((filter_params["a_t"],)) < tuple((a,)):
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]

            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[: filter_params["max_n_holes"]]
                filtered_holes = []

                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params["a_h"]:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        dw, dh = self.level_dim[seg_level]
        img = np.array(
            self.read_region(
                x=0,
                y=0,
                w=dw,
                h=dh,
                level=seg_level,
            )
        )
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring

        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale * scale))
        filter_params = filter_params.copy()
        filter_params["a_t"] = filter_params["a_t"] * scaled_ref_patch_area
        filter_params["a_h"] = filter_params["a_h"] * scaled_ref_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
        if len(contours) == 0:
            print("No contours found")
            self.contours_tissue = []
            self.holes_tissue = []
            return False
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params:
            foreground_contours, hole_contours = _filter_contours(
                contours, hierarchy, filter_params
            )  # Necessary for filtering out artifacts

        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

        # exclude_ids = [0,7,9]
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]

        if len(self.contours_tissue) == 0:
            print("No contours found")
            return False

        return True

    def createPatches_bag_hdf5(
        self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs
    ):
        contours = self.contours_tissue
        contour_holes = self.holes_tissue

        print(
            "Creating patches for: ",
            self.name,
            "...",
        )
        elapsed = time.time()
        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)

            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)

                # empty contour, continue
                except StopIteration:
                    continue

                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path

            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)

        return self.hdf5_file

    def _getPatchGenerator(
        self,
        cont,
        cont_idx,
        patch_level,
        save_path,
        patch_size=256,
        step_size=256,
        custom_downsample=1,
        white_black=True,
        white_thresh=15,
        black_thresh=50,
        contour_fn="four_pt",
        use_padding=True,
    ):
        start_x, start_y, w, h = (
            cv2.boundingRect(cont)
            if cont is not None
            else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        )
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if custom_downsample > 1:
            assert custom_downsample == 2
            target_patch_size = patch_size
            patch_size = target_patch_size * 2
            step_size = step_size * 2
            print(
                "Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(
                    custom_downsample, patch_size, patch_size, target_patch_size, target_patch_size
                )
            )

        patch_downsample = (int(self.level_downsamples[patch_level]), int(self.level_downsamples[patch_level]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        if isinstance(contour_fn, str):
            if contour_fn == "four_pt":
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == "four_pt_hard":
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == "center":
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == "basic":
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1])
            stop_x = min(start_x + w, img_w - ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                if not self.isInContours(
                    cont_check_fn, (x, y), self.holes_tissue[cont_idx], ref_patch_size[0]
                ):  # point not inside contour and its associated holes
                    continue

                count += 1
                # patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                patch_arr = np.array(self.read_region(x, y, patch_size, patch_size, patch_level))
                patch_PIL = Image.fromarray(patch_arr)
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))

                if white_black:
                    if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(
                        np.array(patch_PIL), satThresh=white_thresh
                    ):
                        continue

                patch_info = {
                    "x": x // (patch_downsample[0] * custom_downsample),
                    "y": y // (patch_downsample[1] * custom_downsample),
                    "cont_idx": cont_idx,
                    "patch_level": patch_level,
                    "downsample": self.level_downsamples[patch_level],
                    "downsampled_level_dim": tuple(np.array(self.level_dim[patch_level]) // custom_downsample),
                    "level_dim": self.level_dim[patch_level],
                    "patch_PIL": patch_PIL,
                    "name": self.name,
                    "save_path": save_path,
                }

                yield patch_info

        print("patches extracted: {}".format(count))

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False) > 0:
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype="int32") for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype="int32") for hole in holes] for holes in contours]

    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, **kwargs):
        save_path_hdf5 = os.path.join(save_path, str(self.name) + ".h5")
        print(
            "Creating patches for: ",
            self.name,
            "...",
        )
        elapsed = time.time()
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        attr_dicts = []
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print("Processing contour {}/{}".format(idx, n_contours))

            asset_dict, attr_dict = self.process_contour(
                cont, self.holes_tissue[idx], patch_level, save_path, idx, patch_size, step_size, **kwargs
            )
            attr_dicts.append(attr_dict)
            if len(asset_dict) > 0:
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, mode="w")
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode="a")

        reorg_coord_dict = WholeSlideImage.reorg_coord_attr(attr_dicts)
        attr_dict = {"coords": reorg_coord_dict}
        update_attr_hdf5(save_path_hdf5, attr_dict, mode="a")

        return self.hdf5_file

    @staticmethod
    def reorg_coord_attr(coord_attr_dicts):
        # reorganize list of coord attr_dicts into a single dict based on contour index
        # {'coords': {'cont_idx': 0, 'cont_bbox': , 'additional_keys': value}}
        # into
        # {'coords': 'cont_idx': {0: {'cont_bbox': ,}, 1: {'cont_bbox': ,}, 2: {'cont_bbox': ,}, ...}}, 'additional_keys': value}
        # so collapse cont_idx into a single dict with cont_bbox as key, keeping rest of attr_dict as is

        # first lets get all the keys in the first attr_dict
        all_keys = coord_attr_dicts[0]["coords"].keys()
        cont_keys = [key for key in all_keys if "cont_" in key]
        main_keys = [key for key in all_keys if key not in cont_keys]
        # lets make a new dict with all the keys except ones with cont_ in the name
        # use a placeholder value for keys
        reorg_coord_dict = {key: [] for key in main_keys}
        # fill in values for these additional keys from the first attr_dict
        for key in reorg_coord_dict.keys():
            reorg_coord_dict[key] = coord_attr_dicts[0]["coords"][key]

        # now lets fill in the cont keys
        reorg_coord_dict["cont_meta"] = []

        # cont_meta needs to be an array that looks like this:
        # [[[cont_idx, cont_bbox], [cont_idx, cont_bbox], ...]]
        for attr_dict in coord_attr_dicts:
            cont_idx = attr_dict["coords"]["cont_idx"]
            # combine the tuple with the cont_idx as an np array
            bbox_tuple = attr_dict["coords"]["cont_bbox"]
            cont_meta = (cont_idx,) + bbox_tuple
            reorg_coord_dict["cont_meta"].append(cont_meta)

        # convert cont_meta to numpy array
        reorg_coord_dict["cont_meta"] = np.array(reorg_coord_dict["cont_meta"])

        return reorg_coord_dict

    def process_contour(
        self,
        cont,
        contour_holes,
        patch_level,
        save_path,
        cont_idx,
        patch_size=256,
        step_size=256,
        contour_fn="grid_pt",
        use_padding=False,
        top_left=None,
        bot_right=None,
        **kwargs
    ):

        start_x, start_y, w, h = (
            cv2.boundingRect(cont)
            if cont is not None
            else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        )

        patch_downsample = (int(self.level_downsamples[patch_level]), int(self.level_downsamples[patch_level]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)

        if isinstance(contour_fn, str):
            if contour_fn == "skip":
                cont_check_fn = skip_contour_check()
            elif contour_fn == "grid_pt":
                cont_check_fn = isInContourV3_Grid(contour=cont, patch_size=ref_patch_size[0], grid_size=8)
            elif contour_fn == "four_pt":
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == "four_pt_hard":
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == "center":
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == "basic":
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
        # create ij grid index for coordinates
        i_grid, j_grid = np.meshgrid(np.arange(x_coords.shape[0]), np.arange(x_coords.shape[1]), indexing="ij")
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()
        index_candidates = np.array([i_grid.flatten(), j_grid.flatten()]).transpose()

        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = mp.Pool(num_workers)

        iterable = [
            (coord, idx, contour_holes, ref_patch_size[0], cont_check_fn)
            for (coord, idx) in zip(coord_candidates, index_candidates)
        ]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])

        if len(results) > 0:
            # split results into coordinates and indices
            coords, indices = results[:, 0:1], results[:, 1:2]

            # remove index dimension 1
            indices = indices.squeeze(1)
            coords = coords.squeeze(1)

            print("Extracted {} coordinates".format(len(results)))

            asset_dict = {"coords": coords, "grid_ind": indices}

            attr = {
                "cont_idx": cont_idx,
                "cont_bbox": (start_x, start_y, w, h),
                "patch_size": patch_size,
                "patch_level": patch_level,
                "downsample": self.level_downsamples[patch_level],
                "downsampled_level_dim": tuple(np.array(self.level_dim[patch_level])),
                "level_dim": self.level_dim[patch_level],
                "name": self.name,
                "save_path": save_path,
            }

            attr_dict = {"coords": attr}
            return asset_dict, attr_dict

        else:
            return {}, {}

    @staticmethod
    def process_coord_candidate(coord, idx, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord, idx
        else:
            return None

    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0, 0)):
        print("\ncomputing foreground tissue mask")
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
        contours_tissue, contours_holes = zip(
            *sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True)
        )
        for idx in range(len(contours_tissue)):
            cv2.drawContours(
                image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1
            )

            if use_holes:
                cv2.drawContours(
                    image=tissue_mask,
                    contours=contours_holes[idx],
                    contourIdx=-1,
                    color=(0),
                    offset=offset,
                    thickness=-1,
                )
            # contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)

        tissue_mask = tissue_mask.astype(bool)
        print("detected {}/{} of region as tissue".format(tissue_mask.sum(), tissue_mask.size))
        return tissue_mask

    def visWSI(
        self,
        vis_level=0,
        color=(0, 255, 0),
        hole_color=(0, 0, 255),
        annot_color=(255, 0, 0),
        line_thickness=250,
        max_size=None,
        top_left=None,
        bot_right=None,
        custom_downsample=1,
        view_slide_only=False,
        number_contours=False,
        seg_display=True,
        annot_display=True,
    ):

        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample, 1 / downsample]

        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            x, y = top_left
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        else:
            top_left = (0, 0)
            x, y = top_left
            w, h = self.level_dim[vis_level]

        img = np.array(self.read_region(x, y, w, h, vis_level))

        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(
                        img,
                        self.scaleContourDim(self.contours_tissue, scale),
                        -1,
                        color,
                        line_thickness,
                        lineType=cv2.LINE_8,
                        offset=offset,
                    )

                else:  # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(img, [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

                for holes in self.holes_tissue:
                    cv2.drawContours(
                        img, self.scaleContourDim(holes, scale), -1, hole_color, line_thickness, lineType=cv2.LINE_8
                    )

            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(
                    img,
                    self.scaleContourDim(self.contours_tumor, scale),
                    -1,
                    annot_color,
                    line_thickness,
                    lineType=cv2.LINE_8,
                    offset=offset,
                )

        img = Image.fromarray(img)

        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def visSegmentationMask(
        self, mask_hdf5_path=None, vis_level=-1, mask_id="overlay", blank_canvas=False, background_color=(255, 255, 255)
    ):
        # load segmentation mask results from hdf5 file

        if mask_hdf5_path is not None and os.path.exists(mask_hdf5_path):
            print("loading segmentation mask from hdf5 file")
            f = h5py.File(mask_hdf5_path, "r")
            with h5py.File(mask_hdf5_path, "r") as f:
                mask_patches = f[mask_id][:]
                coords = f["coords"][:]
                # grid_ind = f['grid_ind'][:]

        else:
            print(mask_hdf5_path)
            print("segmentation mask found")
            return

        if vis_level < 0:
            vis_level = self.find_downsample_level(10)

        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample, 1 / downsample]
        patch_size = mask_patches.shape[1:3]

        region_size = self.level_dim[vis_level]
        top_left = (0, 0)
        bot_right = self.level_dim[0]
        w, h = region_size

        patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)

        print("\ncreating segmentation mask for: ")
        print("top_left: ", top_left, "bot_right: ", bot_right)
        print("w: {}, h: {}".format(w, h))
        print("scaled patch size: ", patch_size)

        print("total of {} patches".format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if not blank_canvas:
            # downsample original image and use as canvas
            img = np.array(self.read_region(top_left[0], top_left[1], region_size[0], region_size[1], vis_level))
        else:
            # use blank canvas
            img = np.array(Image.new(size=tuple(region_size), mode="RGB", color=background_color))

        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print("progress: {}/{}".format(idx, len(coords)))

            mask_patch = mask_patches[idx]
            coord = coords[idx]
            # rewrite image patch with mask_patch
            img[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]] = mask_patch

        print("Done")

        img = Image.fromarray(img)
        # w, h = img.size

        # if custom_downsample > 1:
        #     img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        return img

    def visHeatmap(
        self,
        scores,
        coords,
        vis_level=-1,
        top_left=None,
        bot_right=None,
        patch_size=(256, 256),
        blank_canvas=False,
        canvas_color=(220, 20, 50),
        alpha=0.4,
        blur=False,
        overlap=0.0,
        segment=True,
        use_holes=True,
        convert_to_percentiles=False,
        binarize=False,
        thresh=0.5,
        max_size=None,
        custom_downsample=1,
        cmap="coolwarm",
    ):
        """
        Args:
            scores (numpy array of float): Attention scores
            coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
            vis_level (int): WSI pyramid level to visualize
            patch_size (tuple of int): Patch dimensions (relative to lvl 0)
            blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
            canvas_color (tuple of uint8): Canvas color
            alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
            blur (bool): apply gaussian blurring
            overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
            segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that
                            self.contours_tissue and self.holes_tissue are not None
            use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
            convert_to_percentiles (bool): whether to convert attention scores to percentiles
            binarize (bool): only display patches > threshold
            threshold (float): binarization threshold
            max_size (int): Maximum canvas size (clip if goes over)
            custom_downsample (int): additionally downscale the heatmap by specified factor
            cmap (str): name of matplotlib colormap to use
        """

        if vis_level < 0:
            vis_level = self.find_downsample_level(32)

        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample, 1 / downsample]  # Scaling from 0 to desired level

        if len(scores.shape) == 2:
            scores = scores.flatten()

        if binarize:
            if thresh < 0:
                threshold = 1.0 / len(scores)

            else:
                threshold = thresh

        else:
            threshold = 0.0

        ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)

        else:
            region_size = self.level_dim[vis_level]
            top_left = (0, 0)
            bot_right = self.level_dim[0]
            w, h = region_size

        patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)

        print("\ncreating heatmap for: ")
        print("top_left: ", top_left, "bot_right: ", bot_right)
        print("w: {}, h: {}".format(w, h))
        print("scaled patch size: ", patch_size)

        ###### normalize filtered scores ######
        if convert_to_percentiles:
            scores = to_percentiles(scores)

        scores /= 100

        ######## calculate the heatmap of raw attention scores (before colormap)
        # by accumulating scores over overlapped regions ######

        # heatmap overlay: tracks attention score over each pixel of heatmap
        # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score = 1.0
                    count += 1
            else:
                score = 0.0
            # accumulate attention
            overlay[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]] += score
            # accumulate counter
            counter[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]] += 1

        if binarize:
            print("\nbinarized tiles based on cutoff of {}".format(threshold))
            print("identified {}/{} patches as positive".format(count, len(coords)))

        # fetch attended region and average accumulated attention
        zero_mask = counter == 0

        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter
        if blur:
            overlay = cv2.GaussianBlur(overlay, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
            # return Image.fromarray(tissue_mask) # tissue mask

        if not blank_canvas:
            # downsample original image and use as canvas
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # use blank canvas
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

        # return Image.fromarray(img) #raw image

        print("\ncomputing heatmap image")
        print("total of {} patches".format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print("progress: {}/{}".format(idx, len(coords)))

            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:

                # attention block
                raw_block = overlay[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]]

                # image block (either blank canvas or orig image)
                img_block = img[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]].copy()

                # color block (cmap applied to attention block)
                color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)

                if segment:
                    # tissue mask block
                    mask_block = tissue_mask[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]]
                    # copy over only tissue masked portion of color block
                    img_block[mask_block] = color_block[mask_block]
                else:
                    # copy over entire color block
                    img_block = color_block

                # rewrite image block
                img[coord[1] : coord[1] + patch_size[1], coord[0] : coord[0] + patch_size[0]] = img_block.copy()

        # return Image.fromarray(img) #overlay
        print("Done")
        del overlay

        if blur:
            img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

        if alpha < 1.0:
            img = self.block_blending(
                img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024
            )

        img = Image.fromarray(img)
        w, h = img.size

        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        print("\ncomputing blend")
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print("using block size: {} x {}".format(block_size_x, block_size_y))

        shift = top_left  # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                # print(x_start, y_start)

                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))

                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img + block_size_y)
                x_end_img = min(w, x_start_img + block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue
                # print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))

                # 3. fetch blend block and size
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
                blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))
                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))

                # 5. blend color block and canvas block
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(
                    blend_block, alpha, canvas, 1 - alpha, 0, canvas
                )
        return img


if __name__ == "__main__":
    # test WholeSlideImage class
    wsi_path = r"D:\Applikate\Kidney-DeepLearning\DeepLabv3_results\test_images\KID-103-3.tif"
    res_path = r"D:\Applikate\Kidney-DeepLearning\DeepLabv3_results\test_images\output"
    wsi = WholeSlideImage(wsi_path)
    # find downsample level closest to 32x
    seg_level = wsi.find_downsample_level(32)
    # wsi.segmentTissue(seg_level=seg_level)
    # wsi.saveSegmentation(mask_file=os.path.join(res_path, "test_mask.pkl"))

    # mask_path = os.path.join(res_path, '{}_mask.jpg'.format(wsi.name))
    # mask = wsi.visWSI(vis_level=2, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0),
    #   line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
    #   number_contours=True, seg_display=True, annot_display=True)
    # mask.save(mask_path)

    # wsi.createPatches_bag_hdf5(save_path=res_path)
    # wsi.process_contours(save_path=res_path)

    # wsi.initSegmentation(mask_file=os.path.join(res_path, "test_mask.pkl"))

    mask_hdf5_path = os.path.join(res_path, "KID-103-3.h5")

    wsi.visSegmentationMask(mask_hdf5_path=mask_hdf5_path, vis_level=0, mask_id="overlay")
