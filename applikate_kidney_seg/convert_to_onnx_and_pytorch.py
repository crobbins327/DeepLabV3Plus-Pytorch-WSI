import tensorflow as tf

# import segmentation_models as sm
from tensorflow.keras.models import load_model
import onnx
from tf2onnx import convert
import os

from onnx2pytorch import ConvertModel
import torch

# sm.set_framework("tf.keras")
# sm.framework()

# Load the TensorFlow model
SEG_MODEL_PATH = "applikate_kidney_seg/models/KID-seg-resnet34-unet-all_512_10x_v6_LRred-final.h5"
MODEL_NAME = os.path.basename(SEG_MODEL_PATH).split(".")[0]

tf_model = load_model(
    SEG_MODEL_PATH,
    custom_objects={
        "focal_loss_plus_dice_loss": None,
        "iou_score": None,
        "f1-score": None,
    },
)

# tf_model = load_model(
#     SEG_MODEL_PATH,
#     custom_objects={
#         "focal_loss_plus_dice_loss": sm.losses.categorical_focal_dice_loss,
#         "iou_score": sm.metrics.IOUScore(threshold=0.5),
#         "f1-score": sm.metrics.FScore(threshold=0.5),
#     },
# )

# SEG_MODEL_PATH = "applikate_kidney_seg/models/KID-seg-inceptionresnetv2-unet-all_512_10x_v2-3.h5"
# MODEL_NAME = os.path.basename(SEG_MODEL_PATH).split(".")[0]

# tf_model = load_model(
#     SEG_MODEL_PATH,
#     custom_objects={
#         "focal_loss_plus_dice_loss": None,
#         "iou_score": None,
#         "f1-score": None,
#         # "InceptionResNetV2": sm.get_preprocessing("inceptionresnetv2"),  # Provide the backbone here
#     },
# )

# Convert to ONNX
spec = (tf.TensorSpec((None, 512, 512, 3), tf.float32),)  # Define the input shape
onnx_model, _ = convert.from_keras(tf_model, input_signature=spec)

# Save ONNX model
onnx.save(onnx_model, f"applikate_kidney_seg/models/{MODEL_NAME}.onnx")

# Load the ONNX model
# onnx_model_path = "applikate_kidney_seg/models/your_model_name.onnx"
# onnx_model = onnx.load(onnx_model_path)

# Convert ONNX to PyTorch
pytorch_model = ConvertModel(onnx_model)

# Save the PyTorch model if needed
torch.save(pytorch_model.state_dict(), f"applikate_kidney_seg/models/{MODEL_NAME}.pth")

# Optionally: Print the converted model architecture
print(pytorch_model)
