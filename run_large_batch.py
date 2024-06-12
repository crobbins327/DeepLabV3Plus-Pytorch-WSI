import subprocess
import shutil
import os

def run_predict_wsi(input_dir, dataset, model, ckpt_path, separable_conv=False):
    # Build the command line arguments
    args = [
        "python", "predict_wsi.py",
        "--input", input_dir,
        "--dataset", dataset,
        "--model", model,
        "--ckpt", ckpt_path
    ]
    
    if separable_conv:
        args.append("--separable_conv")

    # Set the current working directory and environment variable PYTHONPATH
    # cwd = "C:\\Users\\snibb\\Projects\\Applikate\\Kidney\\DeepLabV3Plus-Pytorch-WSI"  # Update this path to your workspace folder
    # env = {"PYTHONPATH": cwd}

    # Run the subprocess
    result = subprocess.run(args)
    
    # Output the result
    print("Output:", result.stdout)
    print("Errors:", result.stderr)
    print("Return Code:", result.returncode)

# Example usage
batch_cases = [
    "D:\\Applikate\\Kidney-DeepLearning\\KID-MED-027_s50",
    "D:\\Applikate\\Kidney-DeepLearning\\KID-MED-028_S50",
    "D:\\Applikate\\Kidney-DeepLearning\\KID-MED-029",
    "D:\\Applikate\\Kidney-DeepLearning\\KID-MED-030",
    "D:\\Applikate\\Kidney-DeepLearning\\KID-MED-031",
    "D:\\Applikate\\Kidney-DeepLearning\\KID-MED-034",
    "D:\\Applikate\\Kidney-DeepLearning\\KID-MED-036",
    "D:\\Applikate\\Kidney-DeepLearning\\KID-MED-037_s50"
]

# for b in batch_cases:
#     run_predict_wsi(
#         b,
#         "KID-MP-10cell",
#         "deeplabv3plus_resnet101",
#         "D:\\Applikate\\Kidney-DeepLearning\\DeepLabv3_results\\KID-MP-10cell_512_resnet101\\checkpoints\\100k\\best_deeplabv3plus_resnet101_KID-MP-10cell_os16.pth",
#         separable_conv=True
#     )

transfer_dir = "D:\\Applikate\\Kidney-DeepLearning\\kidney-cell-seg"

for b in batch_cases:
    sub_dir = os.path.join(transfer_dir, os.path.basename(b))
    # copy the output folder to the transfer directory
    shutil.copytree(os.path.join(b, "output"), sub_dir)