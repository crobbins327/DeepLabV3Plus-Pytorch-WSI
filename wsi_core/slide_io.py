# Purpose: read slides and metadata from variety of wsi formats

import os
import math

# from skimage import io, transform
import pyvips
import numpy as np
from PIL import Image
import pathlib
import re
import multiprocessing

# from pqdm.threads import pqdm
from scipy import stats
from bs4 import BeautifulSoup
from statistics import mode
import time
import sys
import itertools
import xml.etree.ElementTree as elementTree
import unicodedata
import ome_types

# import jpype
# from aicspylibczi import CziFile
from tqdm import tqdm

# import scyjava
from difflib import get_close_matches
import traceback
from wsi_core import slide_tools

pyvips.cache_set_max(0)

CMAP_AUTO = "auto"
"""
str: Default argument to get channel colors.
"""

MAX_TILE_SIZE = 2**10
"""int: maximum tile used to read or write images"""

# BF_RDR = "bioformats"
# """str: Name of Bioformats reader."""

VIPS_RDR = "libvips"
"""str: Name of pyvips reader"""

OPENSLIDE_RDR = "openslide"
"""str: Name of OpenSlide reader"""

IMG_RDR = "skimage"
"""str: Name of image reader"""

# PIXEL_UNIT = "px"
PIXEL_UNIT = "pixel"
"""str: Physical unit when the unit can't be found in the metadata"""

MICRON_UNIT = "\u00B5m"
"""str: Phyiscal unit for micron/micrometers"""

ALL_OPENSLIDE_READABLE_FORMATS = [".svs", ".tif", ".vms", ".vmu", ".ndpi", ".scn", ".mrxs", ".tiff", ".svslide", ".bif"]
"""list: File extensions that OpenSlide can read"""


# VIPS_READABLE_FORMATS = [".jpg", ".jpeg", ".jpe", ".jif", ".jfif", ".jfi",
#                            ".tif", ".tiff", ".png", ".webp", ".heif", ".heifs",
#                            ".heic", ".heics", ".avci", ".avcs", ".avif", ".hif", ".avif",
#                            ".fits", ".fit", ".fts", ".exr", ".pdf", ".svg", ".hdr",
#                            ".pbm", ".pgm", ".ppm", ".pfm",
#                            ".csv", ".gif", ".img", ".vips",
#                            ".nii", ".nii.gz",
#                            ".dzi" ".xml", ".dcm", ".ome.tiff", ".ome.tif"]

# VIPS_READABLE_FORMATS = pyvips.get_suffixes()
VIPS_READABLE_FORMATS = [*pyvips.get_suffixes(), *ALL_OPENSLIDE_READABLE_FORMATS, ".ome.tiff", ".ome.tif"]
# VIPS_READABLE_FORMATS += [".ome.tiff", ".ome.tif", *ALL_OPENSLIDE_READABLE_FORMATS]
"""list: File extensions that libvips can read. See https://github.com/libvips/libvips
"""

# BF_READABLE_FORMATS = None
# """list: File extensions that Bioformats can read.
#    Filled in after initializing JVM"""

OPENSLIDE_ONLY = None
"""list: File extensions that OpenSlide can read but Bioformats can't.
   Filled in after initializingJVM"""

# FormatTools = None
# """Bioformats FormatTools.
#    Created after initializing JVM"""

# BF_UNIT = None
# """Bioformats UNITS.
#    Created after initializing JVM."""

# BF_MICROMETER = None
# """Bioformats Unit mircometer object.
#    Created after initializing JVM."""

ome = None
"""Bioformats ome from bioforamts_jar.
   Created after initializing JVM."""

loci = None
"""Bioformats loci from bioforamts_jar.
   Created after initializing JVM."""

OME_TYPES_PARSER = "lxml"


# Read slides #
class MetaData(object):
    """Store slide metadata

    To be filled in by a SlideReader object

    Attributes
    ----------

    name : str
        Name of slide.

    series : int
        Series number.

    server : str
        String indicating what was used to read the metadata.

    slide_dimensions :
        Dimensions of all images in the pyramid (width, height).

    is_rgb : bool
        Whether or not the image is RGB.

    pixel_physical_size_xyu :
        Physical size per pixel and the unit.

    channel_names : list
        List of channel names. None if image is RGB

    n_channels : int
        Number of channels.

    original_xml : str
        Xml string created by bio-formats

    bf_datatype : str
        String indicating bioformats image datatype

    optimal_tile_wh : int
        Tile width and height used to open and/or save image

    """

    def __init__(self, name, server, series=0):
        """

        Parameters
        ----------
        name : str
            Name of slide.

        server : str, optional
            String indicating what was used to read the metadata.

        series : int, optional
            Series number.

        """

        self.name = name
        self.series = series
        self.num_series = None
        self.server = server
        self.slide_dimensions = []
        self.slide_downsamples = []
        self.is_rgb = None
        self.pixel_physical_size_xyu = []
        self.channel_names = None
        self.n_channels = 0
        self.n_z = 1
        self.n_t = 1
        self.original_xml = None
        self.bf_datatype = None
        self.bf_pixel_type = None
        self.is_little_endian = None
        self.optimal_tile_wh = 1024


class SlideReader(object):
    """Read slides and get metadata

    Attributes
    ----------
    slide_f : str
        Path to slide

    metadata : MetaData
        MetaData containing some basic metadata about the slide

    series : int
        Image series

    """

    def __init__(self, src_f, *args, **kwargs):
        """
        Parameters
        -----------
        src_f : str
            Path to slide

        """

        self.src_f = str(src_f)
        self.metadata = None
        self.series = 0

    def slide2vips(self, level, xywh=None, *args, **kwargs):
        """Convert slide to pyvips.Image

        Parameters
        -----------
        level : int
            Pyramid level

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        vips_slide : pyvips.Image
            An  of the slide or the region defined by xywh

        """

    def slide2image(self, level, xywh=None, *args, **kwargs):
        """Convert slide to image

        Parameters
        -----------
        level : int
            Pyramid level

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        img : ndarray
            An image of the slide or the region defined by xywh

        """

    def guess_image_type(self):
        f"""Guess if image is {slide_tools.IHC_NAME} or {slide_tools.IF_NAME}

            Brightfield : RGB or uint8 + 3 channels (after removing alpha)
            Immunofluorescence: != 3 channels and not RGB

        Returns
        -------
        img_type : str
            Image type

        """

        if self.metadata.is_rgb:
            img_type = slide_tools.IHC_NAME
        else:
            img_type = slide_tools.IF_NAME

        return img_type

    def scale_physical_size(self, level):
        """Get resolution pyramid level

        Scale resolution to be for requested pyramid level

        Parameters
        ----------
        level : int
            Pyramid level

        Returns
        -------
        level_xy_per_px: tuple

        """

        level_0_shape = self.metadata.slide_dimensions[0]
        level_shape = self.metadata.slide_dimensions[level]
        scale_x = level_0_shape[0] / level_shape[0]
        scale_y = level_0_shape[1] / level_shape[1]

        level_xy_per_px = (
            scale_x * self.metadata.pixel_physical_size_xyu[0],
            scale_y * self.metadata.pixel_physical_size_xyu[1],
            self.metadata.pixel_physical_size_xyu[2],
        )

        return level_xy_per_px

    def create_metadata(self):
        """Create and fill in a MetaData object

        Returns
        -------
        metadata : MetaData
            MetaData object containing metadata about slide

        """

    def get_channel_index(self, channel):

        if isinstance(channel, int):
            matching_channel_idx = channel

        elif isinstance(channel, str):
            cnames = [x.upper() for x in self.metadata.channel_names]
            try:
                best_match = get_close_matches(channel.upper(), cnames)[0]
                matching_channel_idx = cnames.index(best_match)
                if best_match.upper() != channel.upper():
                    msg = f"Cannot find exact match to channel '{channel}' in {slide_tools.get_name(self.src_f)}. Using channel {best_match}"
                    # valtils.print_warning(msg)
                    print(msg)
            except Exception as e:
                traceback_msg = traceback.format_exc()
                matching_channel_idx = 0
                msg = (
                    f"Cannot find channel '{channel}' in {slide_tools.get_name(self.src_f)}."
                    f" Available channels are {self.metadata.channel_names}."
                    f" Using channel number {matching_channel_idx}, which has name {self.metadata.channel_names[matching_channel_idx]}"
                )

                # valtils.print_warning(msg)
                print(msg)

        else:
            matching_channel_idx = 0

        return matching_channel_idx

    def get_channel(self, level, series, channel):
        """Get channel from slide

        Parameters
        ----------
        level : int
            Pyramid level

        series  : int
            Series number

        channel : str, int
            Either the name of the channel (string), or the index of the channel (int)

        Returns
        -------
        img_channel : ndarray
            Specified channel sliced from the slide/image

        """

        matching_channel_idx = self.get_channel_index(channel)
        image = self.slide2image(level=level, series=series)
        img_channel = image[..., matching_channel_idx]

        return img_channel

    def _check_rgb(self, *args, **kwargs):
        """Determine if image is RGB

        Returns
        -------
        is_rgb : bool
            Whether or not the image is RGB

        """

    def _get_channel_names(self, *args, **kwargs):
        """Get names of each channel

        Get list of channel names

        Returns
        -------
        channel_names : list
            List of channel names

        """

    def _get_slide_dimensions(self, *args, **kwargs):
        """Get dimensions of slide at all pyramid levels

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        """

    def _get_pixel_physical_size(self, *args, **kwargs):
        """Get resolution of slide

        Returns
        -------
        res_xyu : tuple
            Physical size per pixel and the unit, e.g. u'\u00B5m'

        Notes
        -----
            If physical unit is micron, it must be u'\u00B5m',
            not mu (u'\u03bcm') or u.

        """


class VipsSlideReader(SlideReader):
    """Read slides using pyvips
    Pyvips includes OpenSlide and so can read those formats as well.

    Attributes
    ----------
    use_openslide : bool
        Whether or not openslide can be used to read this slide.

    is_ome : bool
        Whether ot not the side is an ome.tiff.

    Notes
    -----
    When using openslide, lower levels can only be read without distortion,
    if pixman version 0.40.0 is installed. As of Oct 7, 2021, Macports only has
    pixman version 0.38, which produces distorted lower level images. If using
    macports may need to install from source do  "./configure --prefix=/opt/local/"
    when installing from source.

    """

    def __init__(self, src_f, *args, **kwargs):
        super().__init__(src_f=src_f, *args, **kwargs)
        self.use_openslide = check_to_use_openslide(self.src_f)
        self.is_ome = check_is_ome(self.src_f)
        self.metadata = self.create_metadata()
        self.verify_xml()

    def create_metadata(self):

        if self.use_openslide:
            server = OPENSLIDE_RDR
        else:
            server = VIPS_RDR

        meta_name = f"{os.path.split(self.src_f)[1]}_Series(0)".strip("_")
        slide_meta = MetaData(meta_name, server)
        vips_img = pyvips.Image.new_from_file(self.src_f)

        slide_meta.is_rgb = self._check_rgb(vips_img)
        slide_meta.n_channels = vips_img.bands
        if (slide_meta.is_rgb and vips_img.hasalpha() >= 1) or self.use_openslide:
            # Will remove alpha channel after reading
            slide_meta.n_channels = vips_img.bands - vips_img.hasalpha()

        slide_meta.slide_dimensions = self._get_slide_dimensions(vips_img)
        # get slide downsamples from using first dimension for list of slide dimensions
        slide_meta.slide_downsamples = [1] + [
            round(slide_meta.slide_dimensions[0][0] / x[0]) for x in slide_meta.slide_dimensions[1:]
        ]

        img_xml = self._get_xml(vips_img)
        if img_xml is not None:
            try:
                slide_meta = metadata_from_xml(xml=img_xml, name=slide_meta.name, server=server, metadata=slide_meta)
            except Exception as e:
                # print(f"Can't parse xml for {slide_meta.name} due to error {e}")
                slide_meta = self._get_metadata_vips(slide_meta, vips_img)

        else:
            slide_meta = self._get_metadata_vips(slide_meta, vips_img)

        if slide_meta.is_rgb:
            slide_meta.channel_names = None

        return slide_meta

    def verify_xml(self):
        vips_img = pyvips.Image.new_from_file(self.src_f)
        img_xml = self._get_xml(vips_img)
        if img_xml is not None and not self.use_openslide:
            # Don't check openslide images, as metadata counts alpha channel
            try:
                ome_info = ome_types.from_xml(img_xml, parser=OME_TYPES_PARSER)
                assert len(ome_info.images) > 0

            except Exception as e:
                return None

            read_img = self.slide2vips(0)
            self.metadata = check_xml_img_match(img_xml, read_img, self.metadata, series=self.series)

    def _get_metadata_vips(self, slide_meta, vips_img):
        slide_meta.n_channels = vips_img.bands
        slide_meta.channel_names = self._get_channel_names(vips_img, n_channels=slide_meta.n_channels)
        slide_meta.pixel_physical_size_xyu = self._get_pixel_physical_size(vips_img)
        np_dtype = slide_tools.VIPS_FORMAT_NUMPY_DTYPE[vips_img.format]
        slide_meta.bf_datatype = slide_tools.NUMPY_FORMAT_BF_DTYPE[str(np_dtype().dtype)]
        slide_meta.bf_pixel_type = slide_tools.BF_DTYPE_PIXEL_TYPE[slide_meta.bf_datatype]
        slide_meta.is_little_endian = sys.byteorder.startswith("l")
        slide_meta.original_xml = self._get_xml(vips_img)
        slide_meta.optimal_tile_wh = get_tile_wh(self, 0, get_shape(vips_img)[0:2][::-1])

        return slide_meta

    def _slide2vips_ome_one_series(self, level, *args, **kwargs):
        """Use pyvips to read an ome.tiff image that has only 1 series

        Pyvips throws an error when trying to read other series
        because they may have a different shape than the 1st one
        https://github.com/libvips/pyvips/issues/262

        Parameters
        -----------
        level : int
            Pyramid level

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        vips_slide : pyvips.Image
            An  of the slide or the region defined by xywh

        """

        toilet_roll = pyvips.Image.new_from_file(self.src_f, n=-1, subifd=level - 1)
        page = pyvips.Image.new_from_file(self.src_f, n=1, subifd=level - 1, access="random")
        if page.interpretation == "srgb":
            vips_slide = page
        else:
            page_height = page.height
            pages = [
                toilet_roll.crop(0, y, toilet_roll.width, page_height)
                for y in range(0, toilet_roll.height, page_height)
            ]

            vips_slide = pages[0].bandjoin(pages[1:])
            if vips_slide.bands == 1:
                vips_slide = vips_slide.copy(interpretation="b-w")
            else:
                vips_slide = vips_slide.copy(interpretation="multiband")
                self.metadata.n_channels = vips_slide.bands

        return vips_slide

    def slide2vips(self, level, xywh=None, *args, **kwargs):
        """Convert slide to pyvips.Image

        Parameters
        -----------
        level : int
            Pyramid level

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        vips_slide : pyvips.Image
            An  of the slide or the region defined by xywh

        """

        if self.use_openslide:
            vips_slide = pyvips.Image.new_from_file(self.src_f, level=level, access="random")[0:3]

        elif self.is_ome:
            if self.metadata.num_series == 1:
                vips_slide = self._slide2vips_ome_one_series(level=level, *args, **kwargs)
            else:
                vips_slide = pyvips.Image.new_from_file(self.src_f, subifd=level - 1, access="random")

        else:
            try:
                vips_slide = pyvips.Image.new_from_file(self.src_f, subifd=level - 1, access="random")
            except Exception as e:
                if level > 0 and len(self.metadata.slide_dimensions) > 1:
                    # Pyramid image but each level is a page, not a SubIFD
                    vips_slide = pyvips.Image.new_from_file(self.src_f, page=level, access="random")
                else:
                    # Regular images like png or jpeg don't have SubIFD or pages
                    vips_slide = pyvips.Image.new_from_file(self.src_f, access="random")

        if self.metadata.is_rgb and vips_slide.hasalpha() >= 1:
            # Remove alpha channel
            vips_slide = vips_slide.flatten()

        if xywh is not None:
            vips_slide = vips_slide.extract_area(*xywh)

        return vips_slide

    def slide2image(self, level, xywh=None, *args, **kwargs):
        """Convert slide to image

        Parameters
        -----------
        level : int
            Pyramid level.

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        img : ndarray
            An image of the slide or the region defined by xywh

        """

        vips_slide = self.slide2vips(level=level, xywh=xywh, *args, **kwargs)
        vips_img = slide_tools.vips2numpy(vips_slide)

        return vips_img

    def _check_rgb(self, vips_img):
        """Determine if image is RGB

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide.

        Returns
        -------
        is_rgb : bool
            Whether or not the image is RGB.

        """

        return vips_img.interpretation == "srgb"

    def _get_xml(self, vips_img):
        img_desc = None
        vips_fields = vips_img.get_fields()

        if "openslide.vendor" in vips_fields:
            img_desc = openslide_desc_2_omexml(vips_img)

        elif "image-description" in vips_fields:
            img_desc = vips_img.get("image-description")

        return img_desc

    def _get_channel_names(self, vips_img, *args, **kwargs):
        """Get names of each channel

        Get list of channel names.

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide.

        Returns
        -------
        channel_names : list
            List of channel naames.

        """

        vips_fields = vips_img.get_fields()
        channel_names = None
        if "n-pages" in vips_fields and "image-description" in vips_fields:
            n_pages = vips_img.get("n-pages")
            channel_names = []
            for i in range(n_pages):
                page = pyvips.Image.new_from_file(self.src_f, page=i)
                page_metadata = page.get("image-description")

                # with valtils.HiddenPrints():
                page_soup = BeautifulSoup(page_metadata, features="lxml")

                cname = page_soup.find("name")

                if cname is not None:
                    if cname.text not in channel_names:
                        channel_names.append(cname.text)

        if (channel_names is None or len(channel_names) == 0) and not vips_img.interpretation == "srgb":
            channel_names = get_default_channel_names(vips_img.bands, src_f=self.src_f)
        return channel_names

    def _get_slide_dimensions(self, vips_img):
        """Get dimensions of slide at all pyramid levels using openslide.

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide.

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        """

        if self.use_openslide:
            slide_dimensions = self._get_slide_dimensions_openslide(vips_img)

        elif self.is_ome:
            slide_dimensions = self._get_slide_dimensions_ometiff(vips_img)

        else:
            slide_dimensions = self._get_slide_dimensions_vips(vips_img)

        return slide_dimensions

    def _get_slide_dimensions_ometiff(self, vips_img, *args):

        if "n-subifds" not in vips_img.get_fields():
            return self._get_slide_dimensions_vips(vips_img)

        n_levels = vips_img.get("n-subifds") + 1
        slide_dims_wh = [None] * n_levels
        for i in range(0, n_levels):
            page = pyvips.Image.new_from_file(self.src_f, n=1, subifd=i - 1)
            slide_dims_wh[i] = np.array([page.width, page.height])

        slide_dims_wh = np.array(slide_dims_wh)

        return slide_dims_wh

    def _get_slide_dimensions_openslide(self, vips_img):
        """Get dimensions of slide at all pyramid levels using openslide

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        """

        n_levels = eval(vips_img.get("openslide.level-count"))
        slide_dims = np.array(
            [
                [eval(vips_img.get(f"openslide.level[{i}].width")), eval(vips_img.get(f"openslide.level[{i}].height"))]
                for i in range(n_levels)
            ]
        )

        return slide_dims

    def _get_slide_dimensions_vips(self, vips_img):
        """Get dimensions of slide at all pyramid levels using vips

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        """

        vips_fields = vips_img.get_fields()
        if "n-pages" in vips_fields:
            n_pages = vips_img.get("n-pages")
            all_dims = []
            all_channels = []
            for i in range(n_pages):
                try:
                    page = pyvips.Image.new_from_file(self.src_f, page=i)
                except pyvips.error.Error as e:
                    print(f"error at page {i}: {e}")

                w = page.width
                h = page.height
                c = page.bands

                all_dims.append([w, h])
                all_channels.append(c)

            try:
                most_common_channel_count = stats.mode(all_channels, keepdims=True)[0][0]
            except Exception as e:
                most_common_channel_count = stats.mode(all_channels)[0][0]

            all_dims = np.array(all_dims)
            keep_idx = np.where(all_channels == most_common_channel_count)[0]
            slide_dims = all_dims[keep_idx]

        else:
            slide_dims = [[vips_img.width, vips_img.height]]

        return np.array(slide_dims)

    def _get_pixel_physical_size(self, vips_img):
        """Get resolution of slide

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide

        Returns
        -------
        res_xyu : tuple
            Physical size per pixel and the unit, e.g. u'\u00B5m'

        Notes
        -----
            If physical unit is micron, it must be u'\u00B5m',
            not mu (u'\u03bcm') or u.

        """

        res_xyu = None
        if self.use_openslide:
            x_res = eval(vips_img.get("openslide.mpp-x"))
            y_res = eval(vips_img.get("openslide.mpp-y"))
            vips_img.get("slide-associated-images")
            phys_unit = MICRON_UNIT
        else:
            x_res = vips_img.get("xres")
            y_res = vips_img.get("yres")
            has_units = "resolution-unit" in vips_img.get_fields()
            if x_res != 0 and y_res != 0 and has_units:
                # in vips, x_res and y_res are px/mm (https://www.libvips.org/API/current/VipsImage.html#VipsImage--xres)
                # Need to convert to um/px
                x_res = (1 / x_res) * (10**3)
                y_res = (1 / y_res) * (10**3)
                phys_unit = MICRON_UNIT
            else:
                # Default value is 0, so not provided
                x_res = 1
                y_res = 1
                phys_unit = PIXEL_UNIT

        res_xyu = (x_res, y_res, phys_unit)

        return res_xyu


# will return false if openslide.level-count not in vips fields
def check_to_use_openslide(src_f):
    """Determine if OpenSlide can be used to read the slide

    Parameters
    ----------
    src_f : str
        Path to slide

    Returns
    -------
    use_openslide : bool
        Whether or not OpenSlide can be used to read the slide.
        This can happen if the file format is not readable by
        OpenSlide, or if pyvips wasn't installed with OpenSlide
        support.

    """

    use_openslide = False
    img_format = slide_tools.get_slide_extension(src_f)
    if img_format in ALL_OPENSLIDE_READABLE_FORMATS:
        try:
            vips_img = pyvips.Image.new_from_file(src_f)
            vips_fields = vips_img.get_fields()
            if "openslide.level-count" in vips_fields:
                use_openslide = True
        except pyvips.error.Error as e:
            # traceback_msg = traceback.format_exc()
            # valtils.print_warning(e, traceback_msg=traceback_msg)
            print(f"Error reading {src_f} with pyvips: {e}")

    return use_openslide


def check_is_ome(src_f):
    is_ome = re.search(".ome", src_f) is not None and re.search(".tif*", src_f) is not None
    if is_ome:
        # Verify that image is valid ome.tiff
        try:
            ome_types.from_tiff(src_f)
            is_ome = True
        except Exception as e:
            is_ome = False

    return is_ome


def check_to_use_vips(src_f):

    f_extension = slide_tools.get_slide_extension(src_f)
    can_use_pyvips = f_extension.lower() in VIPS_READABLE_FORMATS

    return can_use_pyvips


def get_default_channel_names(nc, src_f=None):

    if src_f is not None and nc == 1:
        default_channel_names = [slide_tools.get_name(src_f)]
    else:
        default_channel_names = [f"C{i+1}" for i in range(nc)]

    return default_channel_names


def check_channel_names(channel_names, is_rgb, nc, src_f=None):

    if is_rgb:
        return None

    default_channel_names = get_default_channel_names(nc, src_f=src_f)

    if channel_names is None:
        channel_names = []

    if len(channel_names) == 0 and nc > 0:
        updated_channel_names = default_channel_names
    else:
        updated_channel_names = [
            (
                channel_names[i]
                if (channel_names[i] is not None and channel_names[i] != "None")
                else default_channel_names[i]
            )
            for i in range(nc)
        ]

    renamed_channels = set(updated_channel_names) - set(channel_names)
    if len(renamed_channels) > 0:
        msg = f"some non-RGB channel names were `None` or not provided. Renamed channels are: {sorted(list(renamed_channels))}"
        print(msg)

    return updated_channel_names


def metadata_from_xml(xml, name, server, series=0, metadata=None):
    """
    Use ome-types to extract metadata from xml.
    """

    ome_info = ome_types.from_xml(xml, parser=OME_TYPES_PARSER)
    metadata.num_series = len(ome_info.images)
    ome_img = ome_info.images[series]

    if metadata is None:
        metadata = MetaData(name=name, server=server, series=series)

    if ome_img.pixels.big_endian is not None:
        metadata.is_little_endian = ome_img.pixels.big_endian is False

    has_channel_info = len(ome_img.pixels.channels) > 0
    if has_channel_info:
        metadata.is_rgb = (
            ome_img.pixels.channels[0].samples_per_pixel == 3
            and ome_img.pixels.type.value == "uint8"
            and len(ome_img.pixels.channels) == 1
        )
    else:
        # No channel info, so guess based on image shape and datatype
        metadata.is_rgb = ome_img.pixels.type.value == "uint8" and ome_img.pixels.size_c == 3

    if ome_img.pixels.physical_size_x is not None:
        metadata.pixel_physical_size_xyu = (ome_img.pixels.physical_size_x, ome_img.pixels.physical_size_y, MICRON_UNIT)
    else:
        metadata.pixel_physical_size_xyu = (1, 1, PIXEL_UNIT)

    metadata.n_channels = ome_img.pixels.size_c
    metadata.n_z = ome_img.pixels.size_z
    metadata.n_t = ome_img.pixels.size_t
    metadata.original_xml = ome_info.to_xml()
    metadata.bf_datatype = ome_img.pixels.type.value
    metadata.bf_pixel_type = slide_tools.BF_DTYPE_PIXEL_TYPE[metadata.bf_datatype]

    if not metadata.is_rgb:
        if has_channel_info:
            metadata.channel_names = [ome_img.pixels.channels[i].name for i in range(metadata.n_channels)]
            metadata.channel_names = check_channel_names(
                metadata.channel_names, metadata.is_rgb, metadata.n_channels, src_f=name
            )
        else:
            metadata.channel_names = get_default_channel_names(metadata.n_channels, src_f=name)

    return metadata


def check_xml_img_match(xml, vips_img, metadata, series=0):
    """Make sure that provided xml and image match.
    If there is a mismatch (i.e. channel number), the values in the image take precedence
    """
    ome_obj = ome_types.from_xml(xml, parser=OME_TYPES_PARSER)
    if len(ome_obj.images) > 0:
        ome_img = ome_obj.images[series].pixels
        ome_nc = ome_img.size_c
        ome_size_x = ome_img.size_x
        ome_size_y = ome_img.size_y
        ome_dtype = ome_img.type.name.lower()
    else:
        msg = f"ome-xml for {metadata.name} does not contain any metadata for any images"
        # valtils.print_warning(msg)
        print(msg)
        ome_nc = None
        ome_size_x = None
        ome_size_y = None
        ome_dtype = None

    vips_nc = vips_img.bands
    vips_size_x = vips_img.width
    vips_size_y = vips_img.height
    np_dtype = slide_tools.VIPS_FORMAT_NUMPY_DTYPE[vips_img.format]
    vips_bf_dtype = slide_tools.NUMPY_FORMAT_BF_DTYPE[str(np_dtype().dtype)].lower()

    if ome_nc != vips_nc:
        msg = f"For {metadata.name}, the ome-xml states there should be {ome_nc} channel(s), but there is/are only {vips_nc} channel(s) in the image"
        metadata.n_channels = vips_nc
        if ome_nc is not None:
            # valtils.print_warning(msg)
            print(msg)

    if ome_size_x != vips_size_x:
        msg = f"For {metadata.name}, the ome-xml states the width should be {ome_size_x}, but the image has a width of {vips_size_x}"
        if ome_size_x is not None:
            # valtils.print_warning(msg)
            print(msg)

    if ome_size_y != vips_size_y:
        msg = f"For {metadata.name}, the ome-xml states the height should be {ome_size_y}, but the image has a width of {vips_size_y}"
        if ome_size_y is not None:
            # valtils.print_warning(msg)
            print(msg)

    if ome_dtype != vips_bf_dtype:
        msg = f"For {metadata.name}, the ome-xml states the image type should be {ome_dtype}, but the image has type of {vips_bf_dtype}"
        metadata.bf_datatype = vips_bf_dtype
        metadata.bf_pixel_type = slide_tools.BF_DTYPE_PIXEL_TYPE[vips_bf_dtype]

        if ome_dtype is not None:
            # valtils.print_warning(msg)
            print(msg)

    return metadata


def get_tile_wh(reader, level, out_shape_wh):
    """Get tile width and height to write image"""
    default_wh = 1024

    if reader.metadata is None:
        tile_wh = default_wh
    else:
        slide_meta = reader.metadata
        if slide_meta.optimal_tile_wh is None:
            tile_wh = default_wh
        else:
            tile_wh = slide_meta.optimal_tile_wh

    if level != 0:
        down_sampling = np.mean(slide_meta.slide_dimensions[level] / slide_meta.slide_dimensions[0])
        tile_wh = int(np.round(tile_wh * down_sampling))
        tile_wh = tile_wh - (tile_wh % 16)  # Tile shape must be multiple of 16
        if tile_wh < 16:
            tile_wh = 16
        if np.any(np.array(out_shape_wh[0:2]) < tile_wh):
            tile_wh = min(out_shape_wh[0:2])

    return tile_wh


def get_shape(img):
    """Get shape of image (row, col, nchannels)

    Parameters
    ----------

    img : numpy.array, pyvips.Image
        Image to get shape of

    Returns
    -------
    shape_rc : numpy.array
        Number of rows and columns and channels in the image

    """

    if isinstance(img, pyvips.Image):
        shape_rc = np.array([img.height, img.width])
        ndim = img.bands
    else:
        shape_rc = np.array(img.shape[0:2])

        if img.ndim > 2:
            ndim = img.shape[2]
        else:
            ndim = 1

    shape = np.array([*shape_rc, ndim])

    return shape


def get_shape_xyzct(shape_wh, n_channels):
    """Get image shape in XYZCT format

    Parameters
    ----------
    shape_wh : tuple of int
        Width and heigth of image

    n_channels : int
        Number of channels in the image

    Returns
    -------
    xyzct : tuple of int
        XYZCT shape of the image

    """

    xyzct = (*shape_wh, 1, n_channels, 1)
    return xyzct


def openslide_desc_2_omexml(vips_img):
    """Get basic metatad using openslide and convert to ome-xml"""
    assert "openslide.vendor" in vips_img.get_fields(), "image does not appear to be openslide metadata"
    img_shape_wh = get_shape(vips_img)[0:2][::-1]
    x, y, z, c, t = get_shape_xyzct(shape_wh=img_shape_wh, n_channels=vips_img.bands)

    np_dtype = slide_tools.VIPS_FORMAT_NUMPY_DTYPE[vips_img.format]
    bf_datatype = slide_tools.NUMPY_FORMAT_BF_DTYPE[str(np_dtype().dtype)]

    new_img = ome_types.model.Image(
        id="Image:0",
        pixels=ome_types.model.Pixels(
            id="Pixels:0",
            size_x=x,
            size_y=y,
            size_z=z,
            size_c=c,
            size_t=t,
            type=bf_datatype,
            dimension_order="XYZCT",
            physical_size_x=eval(vips_img.get("openslide.mpp-x")),
            physical_size_x_unit=MICRON_UNIT,
            physical_size_y=eval(vips_img.get("openslide.mpp-y")),
            physical_size_y_unit=MICRON_UNIT,
            metadata_only=True,
        ),
    )

    # Should always be rgb, but checking anyway
    is_rgb = vips_img.interpretation == "srgb"
    if is_rgb:
        rgb_channel = ome_types.model.Channel(id="Channel:0:0", samples_per_pixel=3)
        new_img.pixels.channels = [rgb_channel]

    new_ome = ome_types.OME()
    new_ome.images.append(new_img)

    img_xml = new_ome.to_xml()

    return img_xml


def get_slide_reader(src_f, series=0):
    """Get appropriate SlideReader

    If a slide can be read by openslide and bioformats, VipsSlideReader will be used
    because it can be opened as a pyvips.Image.

    Parameters
    ----------
    src_f : str
        Path to slide

    series : int, optional
        The series to be read. If `series` is None, the the `series`
        will be set to the series associated with the largest image.
        In cases where there is only 1 image in the file, `series`
        will be 0.

    Returns
    -------
    reader: SlideReader
        SlideReader class that can read the slide and and convert them to
        images or pyvips.Images at the specified level and series. They
        also contain a `MetaData` object that contains information about
        the slide, like dimensions at each level, physical units, etc...

    Notes
    -----
    pyvips will be used to open ome-tiff images when `series` is 0

    """

    src_f = str(src_f)
    f_extension = slide_tools.get_slide_extension(src_f)
    is_ome_tiff = check_is_ome(src_f)
    is_tiff = re.search(".tif*", f_extension) is not None and not is_ome_tiff
    is_czi = f_extension == ".czi"

    is_flattened_tiff = False
    bf_reads_flat = False
    # if is_tiff:
    # is_flattened_tiff, _ = check_flattened_pyramid_tiff(src_f, check_with_bf=False)[0:2]

    # if series is None:
    #     series = 0

    one_series = True
    if is_ome_tiff:
        ome_obj = ome_types.from_tiff(src_f)
        one_series = len(ome_obj.images) == 1

    can_use_vips = check_to_use_vips(src_f)
    can_use_openslide = check_to_use_openslide(src_f)  # Checks openslide is installed

    # Give preference to vips/openslide since it will be fastest
    if (can_use_vips or can_use_openslide) and series in [0, None] and not is_flattened_tiff:
        return VipsSlideReader

    # if is_czi:
    #     is_jpegxr = check_czi_jpegxr(src_f)
    #     is_m1_mac = valtils.check_m1_mac()
    #     if is_m1_mac and is_jpegxr:
    #         msg = "Will likely be errors using Bioformats to read a JPEGXR compressed CZI on this Apple M1 machine. Will use CziJpgxrReader instead."
    #         return CziJpgxrReader

    # Check to see if Bio-formats will work
    # init_jvm()
    # can_read_meta_bf, can_read_img_bf = check_to_use_bioformats(src_f, series=series)
    # can_use_bf = can_read_meta_bf and can_read_img_bf
    # if is_flattened_tiff:
    #     _, bf_reads_flat = check_flattened_pyramid_tiff(src_f, check_with_bf=True)[0:2]
    #     # Give preference to BioFormatsSlideReader since it will be faster
    #     if bf_reads_flat and can_read_img_bf:
    #         return BioFormatsSlideReader
    #     else:
    #         return FlattenedPyramidReader

    # if is_czi:
    #     if can_read_img_bf:
    #         # Bio-formats should be able to read CZI
    #         return BioFormatsSlideReader
    #     else:
    #         # Bio-formats unable to read CZI. Check if it is due to jpgxr compression
    #         czi = CziFile(src_f)
    #         comp_tree = czi.meta.findall(".//OriginalCompressionMethod")
    #         if len(comp_tree) > 0:
    #             is_czi_jpgxr = comp_tree[0].text.lower() == "jpgxr"
    #         else:
    #             is_czi_jpgxr = False

    #         if is_czi_jpgxr:
    #             return CziJpgxrReader
    #         else:
    #             msg = f"Unable to find reader to open {os.path.split(src_f)[-1]}"
    #             valtils.print_warning(msg, rgb=Fore.RED)

    #             return None

    # if can_use_bf:
    #     return BioFormatsSlideReader

    # # Try using scikit-image. Not ideal if image is large
    # try:
    #     ImageReader(src_f)
    #     return ImageReader
    # except:
    #     pass

    msg = f"Unable to find reader to open {os.path.split(src_f)[-1]}"
    # valtils.print_warning(msg, rgb=Fore.RED)
    print(msg)

    return None


def get_slide(src_f, series=None):
    slide_reader_cls = get_slide_reader(src_f, series=series)
    if slide_reader_cls is None:
        raise ValueError(f"Unable to find reader to open {src_f}")
    return slide_reader_cls(src_f=src_f)
