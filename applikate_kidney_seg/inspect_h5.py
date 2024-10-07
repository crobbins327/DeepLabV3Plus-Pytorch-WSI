import h5py
import os
import tensorflow as tf

# load tf model
SEG_MODEL_PATH = "applikate_kidney_seg/models/KID-seg-resnet34-unet-all_512_10x_v6_LRred-final.h5"
MODEL_NAME = os.path.basename(SEG_MODEL_PATH).split(".")[0]
tf_model = tf.keras.models.load_model(
    SEG_MODEL_PATH,
    custom_objects={
        "focal_loss_plus_dice_loss": None,
        "iou_score": None,
        "f1-score": None,
    },
)

# Load the HDF5 file
h5_file_path = "applikate_kidney_seg/models/512x512_10x_KIDSeg-all-train.h5"
h5_file = h5py.File(h5_file_path, "r")

# List all groups
print("Keys: %s" % h5_file.keys())
a_group_key = list(h5_file.keys())[0]

# Get the data
data = h5_file[a_group_key]
print(data)
print(data.shape)
print(data.dtype)


# test the model

tf_model.summary()

output = 



# Close the file
h5_file.close()
# Output:
# Keys: <KeysViewHDF5 ['X_train', 'X_val', 'Y_train', 'Y_val']>
# <HDF5 dataset "X_train": shape (1000, 512, 512, 3), type "<f4">
# (1000, 512, 512, 3)
# float32
# The HDF5 file contains four datasets: 'X_train', 'X_val', 'Y_train', and 'Y_val'. The 'X_train' dataset has a shape of (1000, 512, 512, 3) and a data type of float32. You can use this information to load and process the data stored in the HDF5 file.
