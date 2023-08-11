from huggingface_hub import snapshot_download
import os
import sys
import shutil
from pathlib import Path


other_models = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp\weights")
src_dir = r"openvino-ai-plugins-gimp\weights"
test_path = os.path.join(other_models, "superresolution-ov")

for folder in os.scandir(src_dir):
    model = os.path.basename(folder)
    model_path = os.path.join(other_models, model)
    if not os.path.isdir(model_path):
        print("Copying {} to {}".format(model, other_models))
        shutil.copytree(Path(folder), model_path)

print("Setup done for superresolution, semantic-segmentation, style-transfer, in-painting") 

print("**** OPENVINO STABLE DIFFUSION 1.4 MODEL SETUP ****") 
choice = input("Do you want to download openvino stable-diffusion-1.4 model? Enter Y/N: ")



if choice == "Y" or choice == "y":

    install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp")
    SD_path = os.path.join(install_location, "weights\stable-diffusion-ov\stable-diffusion-1.4")

    if os.path.isdir(SD_path):
         shutil.rmtree(SD_path)

    repo_id="bes-dev/stable-diffusion-v1-4-openvino"
    download_folder = snapshot_download(repo_id=repo_id, allow_patterns=["*.xml" ,"*.bin"])
    print("download_folder", download_folder)
    shutil.copytree(download_folder, SD_path)
    
                                    
else:
    sys.exit()