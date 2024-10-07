import os
import pyvips
import tifffile
import multiprocessing as mp

Xres = 0.157
Yres = 0.157
Zres = 0.375
magnification = 40.0
# working directory
WORK_DIR = "E:/Applikate/kidney-images/KID-MED-031"
SAVE_DIR = "E:/Applikate/kidney-images/KID-MED-031/ometif"


def convert_tiff_to_ometiff(tiff_file):
    # open the .tiff file
    img = pyvips.Image.new_from_file(os.path.join(WORK_DIR, tiff_file))

    image_save_path = os.path.join(SAVE_DIR, tiff_file.replace(".tif", ".ome.tif"))
    img.set_type(pyvips.GValue.gstr_type, "interpretation", "rgb")
    image_height = img.height
    image_bands = img.bands
    # set minimal OME metadata
    img.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    img.set_type(
        pyvips.GValue.gstr_type,
        "image-description",
        f"""<?xml version="1.0" encoding="UTF-8"?>
                    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
                            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                            xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
                        <Instrument ID="Instrument:0">
                            <Objective ID="Objective:0" NominalMagnification="{magnification}"/>
                        </Instrument>
                        <Image ID="Image:0" Name="Image0">
                            <InstrumentRef ID="Instrument:0"/>
                            <ObjectiveSettings ID="Objective:0"/>
                            <Pixels ID="Pixels:0" 
                            DimensionOrder="XYCZT" 
                            Type="uint8" 
                            PhysicalSizeX="{Xres}" 
                            PhysicalSizeY="{Yres}" 
                            PhysicalSizeZ="{Zres}" 
                            SizeX="{img.width}" 
                            SizeY="{image_height}" 
                            SizeC="3" 
                            SizeZ="1" 
                            SizeT="1" 
                            Interleaved="true"
                            >
                                <Channel ID="Channel:0:0" SamplesPerPixel="3"><LightPath/></Channel>
                                <TiffData IFD="0" PlaneCount="1"/>
                            </Pixels>
                        </Image>
                    </OME>""",
    )
    img.tiffsave(
        image_save_path,
        compression="jpeg",
        Q=95,
        tile=True,
        tile_width=512,
        tile_height=512,
        pyramid=True,
        subifd=True,
        bigtiff=True,
    )


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    # list all .tiff files in the current directory
    tiff_files = [f for f in os.listdir(WORK_DIR) if f.endswith(".tif")]
    max_workers = mp.cpu_count() - 2
    # iterate over all .tiff files with a process pool using starmap
    with mp.Pool(max_workers) as pool:
        pool.map(convert_tiff_to_ometiff, tiff_files)
