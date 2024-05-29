import platform
import distro
import json
import subprocess
import re
from huggingface_hub import snapshot_download
import os
import sys
import shutil
from pathlib import Path
from glob import glob
from openvino.runtime import Core


mode_config_filename =  os.path.join(os.path.dirname(__file__), "model_setup_config.json") 
mode_config = None
with open(mode_config_filename) as f:
    mode_config = json.load(f)
    
other_models = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp", "weights")
src_dir = os.path.join(os.path.dirname(__file__), "weights")
test_path = os.path.join(other_models, "superresolution-ov")

access_token  = None

core = Core()
cpu_type = core.get_property('CPU','full_device_name'.upper())
os_type = platform.system().lower()
npu_driver_version = None
npu_arch = None
linux_kernel_version = None

if "ultra" in cpu_type.lower(): #bypass this test for RVP
    npu_arch = core.get_property('NPU','DEVICE_ARCHITECTURE')
    try:	
        if os_type == "windows":
            npu_devid_selection = mode_config['npu_devid_selection'] 
            #TODO: for future platforms, find "npu_arch" version and add its value in model_setup_config.json 
            npu_devid = npu_devid_selection['windows'][npu_arch]
            command = "get-WmiObject Win32_PnPSignedDriver | Where-Object {$_.DeviceID -like '"+npu_devid+"*'} | Select-Object -ExpandProperty DriverVersion"
            npu_driver_version = subprocess.check_output(['powershell.exe', command], shell=True, universal_newlines=True).rstrip().split(".")
        elif os_type == "linux": 
            linux_distro = distro.id()  # Get Linux distribution name
            os_version = distro.version() # Get Linux OS version      
            os_version = [int(part) for part in os_version.split('.')]
            os_version_min = [22, 4] # Minimum Ubuntu OS version. 
            if linux_distro.lower() == "ubuntu" and os_version >= os_version_min:
                #Get NPU driver version
                command = "dpkg -l | grep NPU  | awk '{print $3}'"
                npu_driver_version = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True).rstrip().split("\n")
                command  = "uname -r"
                #Get Linux kernel version.
                kernel_version = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True).rstrip()
                pattern = r"\d\.\d\.\d"
                npu_driver_version = re.findall(pattern, npu_driver_version[0])[0]  
                pattern = r"\d\.\d"
                linux_kernel_version =re.findall(pattern, kernel_version)[0]
            
            else:
                raise ValueError(f"Unsupported Linux Distro: {linux_distro} and OS Version: {os_version} ; Minimum Ubuntu OS version required: {os_version_min}")
        else:
            raise ValueError(f"Unsupported OS type {os_type}")          
    except Exception as e:
        print(f"Error: {e}")

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

def get_revsion(model_name=None):
    revision = None
    if model_name is not None:
        # Get the revision selection configuration
        revision_config = mode_config['revision_selection'] 
        try: 
            if os_type == "windows":
                if model_name in revision_config:
                    if int(npu_driver_version[3]) < 2016:
                        revision = revision_config[model_name]['windows']['<2016'] + "-" + str(npu_arch)
                    else: 
                        revision = revision_config[model_name]['windows']['default'] + "-" + str(npu_arch)
                else:
                    revision = revision_config['default']
            elif os_type == "linux":
                revision = revision_config[model_name]['linux'][linux_kernel_version][npu_driver_version] + "-" + str(npu_arch)
        except KeyError:
            raise ValueError(f"Configuration mismatch! {os} & npu driver : {npu_driver_version} versions")
    return revision
                
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
        revision = None
        if npu_driver_version is not None:
           revision = get_revsion(repo_id)
           
        while True:
            try:  
                download_folder = snapshot_download(repo_id=repo_id, token=access_token, revision=revision)
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
        revision = None
        if npu_driver_version is not None: 
            revision = get_revsion(repo_id)
        while True:
            try:  
               download_folder = snapshot_download(repo_id=repo_id, token=access_token, revision=revision)
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
    
