import segmentation_models_pytorch as smp
import segmentation_models as sm
import numpy as np
from tensorflow.keras.models import load_model
import os
import torch

# Create a U-Net model with InceptionResNetV2 backbone in PyTorch
pytorch_model = smp.Unet(
    encoder_name="resnet34",  # Use InceptionResNetV2 as the encoder
    encoder_weights=None,  # We'll transfer the weights from the TensorFlow model
    in_channels=3,  # RGB input
    classes=5,  # Number of output classes (adjust if needed)
)

print(pytorch_model)


# Load the TensorFlow model
SEG_MODEL_PATH = "applikate_kidney_seg/models/KID-seg-resnet34-unet-all_512_10x_v6_LRred-final.h5"
# SEG_MODEL_PATH = "applikate_kidney_seg/models/KID-seg-inceptionresnetv2-unet-all_512_10x_v2-3.h5"
MODEL_NAME = os.path.basename(SEG_MODEL_PATH).split(".")[0]
# sm.set_framework("tf.keras")
# sm.framework()
# inet = sm.get_preprocessing("inceptionresnetv2")
# Load the TensorFlow model
tf_model = load_model(
    SEG_MODEL_PATH,
    custom_objects={
        "focal_loss_plus_dice_loss": None,
        "iou_score": None,
        "f1-score": None,
        # "Functional": inet,
    },
)

tf_model.summary()


# Example function to extract the weights for a specific layer
def get_tf_weights(tf_layer):
    weights = tf_layer.get_weights()
    return [np.array(w) for w in weights]


def load_tf_weights_into_pytorch(tf_model, pytorch_model):
    for tf_layer, pytorch_layer in zip(tf_model.layers, pytorch_model.modules()):
        # Handle Conv2D layers
        if isinstance(pytorch_layer, torch.nn.Conv2d):
            tf_weights = get_tf_weights(tf_layer)
            pytorch_layer.weight.data = torch.Tensor(np.transpose(tf_weights[0], (3, 2, 0, 1)))  # Adjust dimensions
            if len(tf_weights) > 1:
                pytorch_layer.bias.data = torch.Tensor(tf_weights[1])

        # Handle BatchNorm layers
        elif isinstance(pytorch_layer, torch.nn.BatchNorm2d):
            tf_weights = get_tf_weights(tf_layer)
            pytorch_layer.weight.data = torch.Tensor(tf_weights[0])  # gamma
            pytorch_layer.bias.data = torch.Tensor(tf_weights[1])  # beta
            pytorch_layer.running_mean = torch.Tensor(tf_weights[2])  # moving_mean
            pytorch_layer.running_var = torch.Tensor(tf_weights[3])  # moving_variance


load_tf_weights_into_pytorch(tf_model, pytorch_model)

# Save the PyTorch model
torch.save(pytorch_model.state_dict(), f"applikate_kidney_seg/models/{MODEL_NAME}.pth")

# test the model
print(pytorch_model)
pytorch_model.eval()
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    out = pytorch_model(x)
