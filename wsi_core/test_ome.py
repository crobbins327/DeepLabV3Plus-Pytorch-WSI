import pyvips
import os

root_dir = "D:\Applikate\Kidney-DeepLearning"

im = pyvips.Image.new_from_file(os.path.join(root_dir, "004_HE.ome.tif"), subifd=0)

print(im.get_fields())