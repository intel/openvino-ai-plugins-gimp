from huggingface_hub import snapshot_download
import os
import sys
import shutil
from pathlib import Path
from glob import glob


other_models = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp", "weights")
src_dir = os.path.join("openvino-ai-plugins-gimp", "weights")
test_path = os.path.join(other_models, "superresolution-ov")

access_token  = None

for folder in os.scandir(src_dir):
    model = os.path.basename(folder)
    model_path = os.path.join(other_models, model)
    if not os.path.isdir(model_path):
        print("Copying {} to {}".format(model, other_models))
        shutil.copytree(Path(folder), model_path)

print("Setup done for superresolution, semantic-segmentation, style-transfer, in-painting") 
print("**** OPENVINO STABLE DIFFUSION 1.4 MODEL SETUP ****")
choice = input("Do you want to download openvino stable-diffusion-1.4 model? Enter Y/N: ")

install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp", "weights")

if choice == "Y" or choice == "y":
    SD_path = os.path.join(install_location, "stable-diffusion-ov", "stable-diffusion-1.4")

    if os.path.isdir(SD_path):
        
         shutil.rmtree(SD_path)

    repo_id="bes-dev/stable-diffusion-v1-4-openvino"
    
    while True:
        try:
            download_folder = snapshot_download(repo_id=repo_id, allow_patterns=["*.xml" ,"*.bin"])
            break
        except Exception as e:
             print("Error retry:" + str(e))
    
    shutil.copytree(download_folder, SD_path)
    delete_folder = os.path.join(download_folder, "..", "..", "..")
    shutil.rmtree(delete_folder, ignore_errors=True)

install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp", "weights", "stable-diffusion-ov")

def download_quantized_models(repo_id, model_fp16, model_int8):
    download_flag = True
    
    SD_path_FP16 = os.path.join(install_location, model_fp16)
    
    if os.path.isdir(SD_path_FP16):
            choice = input(f"{repo_id} model folder exist. Do you wish to re-download this model? Enter Y/N: ")
            if choice == "Y" or choice == "y":
                download_flag = True
                shutil.rmtree(SD_path_FP16)
            
            else:
                download_flag = False
                print("%s download skipped",repo_id)
                
    if  download_flag:               
    
        while True:
                try:  
                    download_folder = snapshot_download(repo_id=repo_id, token=access_token)
                    break
                except Exception as e:
                     print("Error retry:" + str(e))
          
            #print("download_folder", download_folder)

        FP16_model = os.path.join(download_folder, "FP16")
        # on some systems, the FP16 subfolder is not created resulting in a installation crash 
        if not os.path.isdir(FP16_model):
            os.mkdir(FP16_model)
        shutil.copytree(download_folder, SD_path_FP16, ignore=shutil.ignore_patterns('FP16', 'INT8'))  
        shutil.copytree(FP16_model, SD_path_FP16, dirs_exist_ok=True)        



    
        if model_int8:
            SD_path_INT8 = os.path.join(install_location, model_int8)           
            
            if os.path.isdir(SD_path_INT8):
                    shutil.rmtree(SD_path_INT8)

            INT8_model = os.path.join(download_folder, "INT8")
            shutil.copytree(download_folder, SD_path_INT8, ignore=shutil.ignore_patterns('FP16', 'INT8'))  
            shutil.copytree(INT8_model, SD_path_INT8, dirs_exist_ok=True)


            delete_folder=os.path.join(download_folder, "..", "..", "..")
            shutil.rmtree(delete_folder, ignore_errors=True)
    
def download_model(repo_id, model_1, model_2):
    download_flag = True

    if "sd-2.1" in repo_id:
        sd_model_1 = os.path.join(install_location, "stable-diffusion-2.1", model_1)
        
    else:        
        sd_model_1 = os.path.join(install_location, "stable-diffusion-1.5", model_1)

    if os.path.isdir(sd_model_1):
        choice = input(f"{repo_id} model folder exist. Do you wish to re-download this model? Enter Y/N: ")
        if choice == "Y" or choice == "y":
            download_flag = True
            shutil.rmtree(sd_model_1)
        else:
            download_flag = False
            print("%s download skipped",repo_id)
                           

    if download_flag:
        while True:
            try:
                download_folder = snapshot_download(repo_id=repo_id, token=access_token)
                break
            except Exception as e:
                print("Error retry:" + str(e))
        if repo_id == "Intel/sd-1.5-lcm-openvino":
            download_model_1 = download_folder
        else:
            download_model_1 = os.path.join(download_folder, model_1) 
        shutil.copytree(download_model_1, sd_model_1)  
         
        if model_2:
            if "sd-2.1" in repo_id:
                sd_model_2 = os.path.join(install_location, "stable-diffusion-2.1", model_2)
            else: 
                sd_model_2 = os.path.join(install_location, "stable-diffusion-1.5", model_2)
            if os.path.isdir(sd_model_2):
                    shutil.rmtree(sd_model_2)
            download_model_2 = os.path.join(download_folder, model_2)
            shutil.copytree(download_model_2, sd_model_2)

        delete_folder=os.path.join(download_folder, "../../..")
        shutil.rmtree(delete_folder, ignore_errors=True)
    
def dl_sd_15_square():
    print("Downloading Intel/sd-1.5-square-quantized Models")
    repo_id = "Intel/sd-1.5-square-quantized"
    model_fp16 = os.path.join("stable-diffusion-1.5", "square")
    model_int8 = os.path.join("stable-diffusion-1.5", "square_int8")
    download_quantized_models(repo_id, model_fp16, model_int8)

def dl_sd_21_square():
    print("Downloading Intel/sd-2.1-square-quantized Models")
    repo_id = "Intel/sd-2.1-square-quantized"
    model_1 = "square"
    model_2 = "square_base"
    download_model(repo_id, model_1, model_2)

def dl_sd_15_portrait():
    print("Downloading Intel/sd-1.5-portrait-quantized Models")
    repo_id = "Intel/sd-1.5-portrait-quantized"
    model_1 = "portrait"
    model_2 = "portrait_512x768"
    download_model(repo_id, model_1, model_2)

def dl_sd_15_landscape():
    print("Downloading Intel/sd-1.5-landscape-quantized Models")
    repo_id = "Intel/sd-1.5-landscape-quantized"
    model_1 = "landscape"
    model_2 = "landscape_768x512"
    download_model(repo_id, model_1, model_2)

def dl_sd_15_inpainting():
    print("Downloading Intel/sd-1.5-inpainting-quantized Models")
    repo_id = "Intel/sd-1.5-inpainting-quantized"
    model_fp16 = "stable-diffusion-1.5-inpainting"
    model_int8 = "stable-diffusion-1.5-inpainting-int8"
    download_quantized_models(repo_id, model_fp16, model_int8)

def dl_sd_15_openpose():
    print("Downloading Intel/sd-1.5-controlnet-openpose-quantized Models")
    repo_id="Intel/sd-1.5-controlnet-openpose-quantized"
    model_fp16 = "controlnet-openpose"
    model_int8 = "controlnet-openpose-int8"
    download_quantized_models(repo_id, model_fp16,model_int8)

def dl_sd_15_canny():
    print("Downloading Intel/sd-1.5-controlnet-canny-quantized Models")
    repo_id = "Intel/sd-1.5-controlnet-canny-quantized"
    model_fp16 = "controlnet-canny"
    model_int8 = "controlnet-canny-int8"
    download_quantized_models(repo_id, model_fp16, model_int8)

def dl_sd_15_scribble():
    print("Downloading Intel/sd-1.5-controlnet-scribble-quantized Models")
    repo_id = "Intel/sd-1.5-controlnet-scribble-quantized"
    model_fp16 = "controlnet-scribble"
    model_int8 = "controlnet-scribble-int8"
    download_quantized_models(repo_id, model_fp16, model_int8)
def dl_sd_15_LCM():
    print("Downloading Intel/sd-1.5-lcm-openvino")
    repo_id = "Intel/sd-1.5-lcm-openvino"
    model_1 = "square_lcm"
    model_2 = None
    download_model(repo_id, model_1, model_2)

def dl_sd_15_Referenceonly():
    print("Downloading Intel/sd-reference-only")
    repo_id = "Intel/sd-reference-only"
    model_fp16 = "controlnet-referenceonly"
    model_int8 = None
    download_quantized_models(repo_id, model_fp16, model_int8)

def dl_all():
    dl_sd_15_square()
    dl_sd_15_portrait()
    dl_sd_15_landscape()
    dl_sd_15_inpainting()
    dl_sd_15_openpose()
    dl_sd_15_canny()
    dl_sd_15_scribble()
    dl_sd_15_LCM()
    dl_sd_15_Referenceonly()
    #dl_sd_21_square()

while True:
    print("=========Chose SD models to download =========")
    print("1 - SD-1.5 Square (512x512)")
    print("2 - SD-1.5 Portrait")
    print("3 - SD-1.5 Landscape")
    print("4 - SD-1.5 Inpainting (512x512 output image)")
    print("5 - SD-1.5 Controlnet-Openpose")
    print("6 - SD-1.5 Controlnet-CannyEdge")
    print("7 - SD-1.5 Controlnet-Scribble")
    print("8 - SD-1.5 LCM ")
    print("9 - SD-1.5 Controlnet-ReferenceOnly")
#    print("10 - SD-2.1 Square (768x768)")
    print("12 - All the above models")
    print("0 - Exit SD Model setup")

    choice = input("Enter the Number for the model you want to download: ")

    if choice=="1":  dl_sd_15_square()
    if choice=="2":  dl_sd_15_portrait()
    if choice=="3":  dl_sd_15_landscape()
    if choice=="4":  dl_sd_15_inpainting()
    if choice=="5":  dl_sd_15_openpose()
    if choice=="6":  dl_sd_15_canny()
    if choice=="7":  dl_sd_15_scribble()
    if choice=="8":  dl_sd_15_LCM()
    if choice=="9":  dl_sd_15_Referenceonly()
    #if choice=="10":  dl_sd_21_square()

    if choice=="12":
        dl_all()
        print("Complete downloaing all models. Exiting SD-1.5 Model setup.........")
        break
    
    if choice=="0":
        print("Exiting SD Model setup.........")
        break
    
