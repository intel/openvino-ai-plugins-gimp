from huggingface_hub import snapshot_download
import os
import sys
import shutil
from pathlib import Path


other_models = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp\weights")
src_dir = r"openvino-ai-plugins-gimp\weights"
test_path = os.path.join(other_models, "superresolution-ov")

access_token  = "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"

for folder in os.scandir(src_dir):
    model = os.path.basename(folder)
    model_path = os.path.join(other_models, model)
    if not os.path.isdir(model_path):
        print("Copying {} to {}".format(model, other_models))
        shutil.copytree(Path(folder), model_path)

print("Setup done for superresolution, semantic-segmentation, style-transfer, in-painting") 

print("**** OPENVINO STABLE DIFFUSION 1.4 MODEL SETUP ****") 
choice = input("Do you want to download openvino stable-diffusion-1.4 model? Enter Y/N: ")

install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp\weights")

if choice == "Y" or choice == "y":

    
    SD_path = os.path.join(install_location, "stable-diffusion-ov\stable-diffusion-1.4")

    if os.path.isdir(SD_path):
         shutil.rmtree(SD_path)

    repo_id="bes-dev/stable-diffusion-v1-4-openvino"
    download_folder = snapshot_download(repo_id=repo_id, allow_patterns=["*.xml" ,"*.bin"])
    print("download_folder", download_folder)
    shutil.copytree(download_folder, SD_path)
  

install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp\weights\stable-diffusion-ov")

def download_hf_model(repo_id, model_fp16, model_int8):



    download_folder = snapshot_download(repo_id=repo_id, token=access_token)
    SD_path_FP16 = os.path.join(install_location, model_fp16)
    
    if os.path.isdir(SD_path_FP16):
            shutil.rmtree(SD_path_FP16)
  
    print("download_folder", download_folder)

    FP16_model = os.path.join(download_folder, "FP16")
    shutil.copytree(download_folder, SD_path_FP16, ignore=shutil.ignore_patterns('FP16', 'INT8'))    
    shutil.copytree(FP16_model, SD_path_FP16)    
    
    if model_int8:
        SD_path_INT8 = os.path.join(install_location, model_int8)

        if os.path.isdir(SD_path_INT8):
            shutil.rmtree(SD_path_INT8)
        
        
        INT8_model = os.path.join(download_folder, "INT8")
        shutil.copytree(download_folder, SD_path_INT8, ignore=shutil.ignore_patterns('FP16', 'INT8'))        
        shutil.copytree(INT8_model, SD_path_INT8)


    delete_folder=os.path.join(download_folder, "../../..")
    shutil.rmtree(delete_folder)  
    
def download_model(repo_id, model_1, model_2):
    download_folder = snapshot_download(repo_id=repo_id, token=access_token)
    sd_model_1 = os.path.join(install_location, "stable-diffusion-1.5", model_1)
    sd_model_2 = os.path.join(install_location, "stable-diffusion-1.5", model_2)
    
    if os.path.isdir(sd_model_1):
            shutil.rmtree(sd_model_1)
    if os.path.isdir(sd_model_2):
            shutil.rmtree(sd_model_2)  

    download_model_1 = os.path.join(download_folder, model_1)  
    download_model_2 = os.path.join(download_folder, model_2)
    shutil.copytree(download_model_1, sd_model_1)  
    shutil.copytree(download_model_2, sd_model_2) 
    
    delete_folder=os.path.join(download_folder, "../../..")
    shutil.rmtree(delete_folder)
    

while True:

    print("=========Chose SD-1.5 models to download =========")
    print("1 - SD-1.5 Square (512x512)")
    print("2 - SD-1.5 Landscape")
    print("3 - SD-1.5 Portrait")
    print("4 - SD-1.5 Inpainting (512x512 output image)")
    print("5 - SD-1.5 Controlnet-Openpose")
    print("6 - SD-1.5 Controlnet-CannyEdge")
    print("7 - SD-1.5 Controlnet-Scribble")
    print("8 - LCM ")
    
    #print("12 - All the above models")
    print("9 - Exit SD-1.5 Model setup")
    

    choice = input("Enter the Number for the model you want to download: ")

    if choice=="1":
        print("Downloading Intel/sd-1.5-square-quantized Models")

        repo_id="Intel/sd-1.5-square-quantized"
        model_fp16 = "stable-diffusion-1.5\square"
        model_int8 = "stable-diffusion-1.5\square_int8"
        download_hf_model(repo_id, model_fp16,model_int8)
        
    if choice=="2":
        print("Downloading Intel/sd-1.5-portrait-quantized Models")

        repo_id="Intel/sd-1.5-portrait-quantized"
        model_1 = "portrait"
        model_2 = "portrait_512x768"
        download_model(repo_id, model_1,model_2)   

    if choice=="3":
        print("Downloading Intel/sd-1.5-landscape-quantized Models")

        repo_id="Intel/sd-1.5-landscape-quantized"
        model_1 = "landscape"
        model_2 = "landscape_768x512"
        download_model(repo_id, model_1,model_2)             
        
    if choice=="4":
        print("Downloading Intel/sd-1.5-inpainting-quantized Models")

        repo_id="Intel/sd-1.5-inpainting-quantized"
        model_fp16 = "stable-diffusion-1.5-inpainting"
        model_int8 = "stable-diffusion-1.5-inpainting-int8"
        download_hf_model(repo_id, model_fp16,model_int8)

    elif choice=="5":
        print("Downloading Intel/sd-1.5-controlnet-openpose-quantized Models")

        repo_id="Intel/sd-1.5-controlnet-openpose-quantized"
        model_fp16 = "controlnet-openpose"
        model_int8 = "controlnet-openpose-int8"
        download_hf_model(repo_id, model_fp16,model_int8)
    elif choice=="6":
        print("Downloading Intel/sd-1.5-controlnet-canny-quantized Models")

        repo_id="Intel/sd-1.5-controlnet-canny-quantized"
        model_fp16 = "controlnet-canny"
        model_int8 = "controlnet-canny-int8"
        download_hf_model(repo_id, model_fp16,model_int8)        


    elif choice=="7":
        print("Downloading Intel/sd-1.5-controlnet-scribble-quantized Models")

        repo_id="Intel/sd-1.5-controlnet-scribble-quantized"
        model_fp16 = "controlnet-scribble"
        model_int8 = "controlnet-scribble-int8"
        download_hf_model(repo_id, model_fp16,model_int8)  
        
    elif choice=="13":
         print("Downloading all the models")
         download_hf_model("Intel/stable-diffusion-1.5-quantized", "stable-diffusion-1.5\square","stable-diffusion-1.5\square_int8")
         download_hf_model("Intel/sd-1.5-controlnet-openpose-quantized", "controlnet-openpose","controlnet-openpose-int8")
         break
    
    elif choice=="9":
        print("Exiting SD-1.5 Model setup.........")
        break
    
