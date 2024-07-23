import platform
import json
import subprocess
import re
from huggingface_hub import snapshot_download
import os
import shutil
from pathlib import Path
from openvino.runtime import Core
import io
import logging

logging.basicConfig(level=logging.INFO)

    
base_model_dir = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp", "weights")
install_location = os.path.join(base_model_dir, "stable-diffusion-ov")
src_dir = os.path.join(os.path.dirname(__file__), "weights")
test_path = os.path.join(install_location, "superresolution-ov")

access_token = None

# Initialize OpenVINO Core
core = Core()
os_type = platform.system().lower()
available_devices = core.get_available_devices()
npu_arch = None
if 'NPU' in available_devices:
    npu_arch = "3720" if "3720" in core.get_property('NPU', 'DEVICE_ARCHITECTURE') else None


def load_model(self, model, model_name, device):
    print(f"Loading {model_name} to {device}")
    start_t = time.time()
    cmmodel =  core.compile_model(os.path.join(model, f"{model_name}.xml"), device)
    print(f"Model Load completed in {time.time() - start_t} seconds")

def install_base_models():
    for folder in os.scandir(src_dir):
        model = os.path.basename(folder)
        model_path = os.path.join(base_model_dir, model)
        if not os.path.isdir(model_path):
            print("Copying {} to {}".format(model, base_model_dir))
            shutil.copytree(Path(folder), model_path)
            
    print("Setup done for superresolution and semantic-segmentation") 
                
def download_quantized_models(repo_id, model_fp16, model_int8):
    download_flag = True
    SD_path_FP16 = os.path.join(install_location, model_fp16)
    SD_path_INT8 = os.path.join(install_location, model_int8)
               
    if os.path.isdir(SD_path_FP16):
            choice = input(f"{repo_id} model folder exist. Do you wish to re-download this model? Enter Y/N: ")
            if choice == "Y" or choice == "y":
                shutil.rmtree(SD_path_FP16)
            else:
                download_flag = False
                print(f"{repo_id} download skipped")
                return download_flag
                
    if  download_flag:               
        while True:
            try:  
                download_folder = snapshot_download(repo_id=repo_id, token=access_token)
                break
            except Exception as e:
                print("Error retry:" + str(e))
 
        FP16_model = os.path.join(download_folder, "FP16")
        # on some systems, the FP16 subfolder is not created resulting in a installation crash 
        if not os.path.isdir(FP16_model):
            os.mkdir(FP16_model)
        shutil.copytree(download_folder, SD_path_FP16, ignore=shutil.ignore_patterns('FP16', 'INT8'))  
        shutil.copytree(FP16_model, SD_path_FP16, dirs_exist_ok=True)       

        if model_int8:            
            if os.path.isdir(SD_path_INT8):
                    shutil.rmtree(SD_path_INT8)

            INT8_model = os.path.join(download_folder, "INT8")
            shutil.copytree(download_folder, SD_path_INT8, ignore=shutil.ignore_patterns('FP16', 'INT8'))  
            shutil.copytree(INT8_model, SD_path_INT8, dirs_exist_ok=True)
            
            delete_folder=os.path.join(download_folder, "..", "..", "..")
            shutil.rmtree(delete_folder, ignore_errors=True)

    return download_flag

def download_model(repo_id, model_1, model_2):
    download_flag = True
    
    if "sd-2.1" in repo_id:
        sd_model_1 = os.path.join(install_location, "stable-diffusion-2.1", model_1)    
    else:        
        sd_model_1 = os.path.join(install_location, "stable-diffusion-1.5", model_1)

    if os.path.isdir(sd_model_1):
        choice = input(f"{repo_id} model folder exist. Do you wish to re-download this model? Enter Y/N: ")
        if choice == "Y" or choice == "y":
            shutil.rmtree(sd_model_1)
        else:
            download_flag = False
            print(f"{repo_id} download skipped")
            return download_flag
                           
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
    
    return download_flag



def compile_and_export_model(core, model_path, output_path, device='NPU', config=None):
    """
    Compile the model and export it to the specified path.
    """
    model = core.compile_model(model_path, device, config=config)
    with io.BytesIO() as model_blob:
        model.export_model(model_blob)
        with open(output_path, 'wb') as f:
            f.write(model_blob.getvalue())

def dl_sd_15_square():
    print("Downloading Intel/sd-1.5-square-quantized Models")
    repo_id = "Intel/sd-1.5-square-quantized"
    model_fp16 = os.path.join("stable-diffusion-1.5", "square")
    model_int8 = os.path.join("stable-diffusion-1.5", "square_int8")
    compile_models = download_quantized_models(repo_id, model_fp16, model_int8)
    
    if npu_arch is not None:
        if not compile_models:
            user_input = input("Do you want to reconfigure models for NPU? Enter Y/N: ").strip().lower()
            if user_input == "y":
                compile_models = True
    
        if compile_models:
            if npu_arch == "3720":
                models_to_compile = [ "text_encoder", "unet_int8"]
                shared_models = ["text_encoder.blob"]
            else:
                models_to_compile = [ "text_encoder", "unet_bs1" , "unet_int8", "vae_encoder" , "vae_decoder" ]
                shared_models = ["text_encoder.blob", "vae_encoder.blob", "vae_decoder.blob"]
    
            for model_name in models_to_compile:
                model_path_fp16 = os.path.join(install_location, model_fp16, model_name + ".xml")
                output_path_fp16 = os.path.join(install_location, model_fp16, model_name + ".blob")
        
                if "unet_int8" in model_name:
                    model_path_int8 = os.path.join(install_location, model_int8, model_name + ".xml")
                    output_path_int8 = os.path.join(install_location, model_int8, model_name + ".blob")
                    print(f"Creating NPU model for {model_name} - INT8")
                    config = {"NPU_DPU_GROUPS":"2"}
                    compile_and_export_model(core, model_path_int8, output_path_int8, config=config)
                else:
                    print(f"Creating NPU model for {model_name}")
                    compile_and_export_model(core, model_path_fp16, output_path_fp16)

            # Copy shared models to INT8 directory
            for blob_name in shared_models:
                shutil.copy(
                    os.path.join(install_location, model_fp16, blob_name),
                    os.path.join(install_location, model_int8, blob_name)
                )

def dl_sd_14_square():
    SD_path = os.path.join(install_location, "stable-diffusion-1.4")
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
    model_fp16 = os.path.join("stable-diffusion-1.5", "inpainting")
    model_int8 = os.path.join("stable-diffusion-1.5", "inpainting_int8")
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
    compile_models = download_model(repo_id, model_1, model_2)
    
    if npu_arch is not None:
        if not compile_models:
            user_input = input("Do you want to reconfigure models for NPU? Enter Y/N: ").strip().lower()
            if user_input == "y":
                compile_models = True
    
        if compile_models:
            if npu_arch == "3720":
                models_to_compile = [ "text_encoder", "unet" ]
            else:
                models_to_compile = [ "text_encoder", "unet" , "vae_decoder" ]
        
            for model_name in models_to_compile:
                model_path = os.path.join(install_location, "stable-diffusion-1.5", model_1, model_name + ".xml")
                output_path = os.path.join(install_location,"stable-diffusion-1.5", model_1, model_name + ".blob")
                print(f"Creating NPU model for {model_name}")
                compile_and_export_model(core, model_path, output_path)
    

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
    dl_sd_21_square()
    

def show_menu():
    """
    Display the menu options for downloading models.
    """
    print("=========Choose SD models to download=========")
    print("1  - SD-1.5 Square (512x512)")
    print("2  - SD-1.5 Portrait")
    print("3  - SD-1.5 Landscape")
    print("4  - SD-1.5 Inpainting (512x512 output image)")
    print("5  - SD-1.5 Controlnet-Openpose")
    print("6  - SD-1.5 Controlnet-CannyEdge")
    print("7  - SD-1.5 Controlnet-Scribble")
    print("8  - SD-1.5 LCM")
    print("9  - SD-1.5 Controlnet-ReferenceOnly")
    print("10 - SD-2.1 Square (768x768)")
    print("11 - SD 1.4 Square")
    print("12 - All the above models")
    print("0  - Exit SD Model setup")

def main():
    install_base_models()
    download_functions = {
                "1": dl_sd_15_square,
                "2": dl_sd_15_portrait,
                "3": dl_sd_15_landscape,
                "4": dl_sd_15_inpainting,
                "5": dl_sd_15_openpose,
                "6": dl_sd_15_canny,
                "7": dl_sd_15_scribble,
                "8": dl_sd_15_LCM,
                "9": dl_sd_15_Referenceonly,
                "10": dl_sd_21_square,
                "11" : dl_sd_14_square,
                "12" : dl_all,
                "0": exit,
            }
    
    while True:
        show_menu()
        choice = input("Enter the number for the model you want to download.\nSpecify multiple options using spaces: ")

        choices = choice.split(" ")
        for ch in choices:
            func = download_functions.get(ch.strip())
            if ch == "0":
                print("Exiting Model setup...")
            if func:
                func()
            else:
                print(f"Invalid choice: {ch.strip()}")

if __name__ == "__main__":
    main()