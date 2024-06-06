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

access_token  = "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" # remove me later, eh? 

core = Core()
cpu_type = core.get_property('CPU','full_device_name'.upper())
os_type = platform.system().lower()
npu_driver_version = None
npu_arch = None
npu_devid_selection= None
linux_kernel_version = None

# Constants
MIN_UBUNTU_VERSION = [22, 4]
LINUX_NPU_COMMAND = "dpkg -l | grep NPU  | awk '{print $3}'"
LINUX_KERNEL_COMMAND = "uname -r"
WINDOWS_NPU_COMMAND_TEMPLATE = "get-WmiObject Win32_PnPSignedDriver | Where-Object {{$_.DeviceID -like '{npu_devid}*'}} | Select-Object -ExpandProperty DriverVersion"

def get_windows_npu_info(npu_arch, npu_devid_selection):
    npu_devid = npu_devid_selection['windows'][npu_arch]
    command = WINDOWS_NPU_COMMAND_TEMPLATE.format(npu_devid=npu_devid)
    return subprocess.check_output(['powershell.exe', command], shell=True, universal_newlines=True).rstrip().split(".")

def get_linux_npu_info():
    linux_distro = distro.id().lower()
    os_version = list(map(int, distro.version().split('.')))
    
    if linux_distro != "ubuntu" or os_version < MIN_UBUNTU_VERSION:
        raise ValueError(f"Unsupported Linux Distro: {linux_distro} and OS Version: {os_version}; Minimum Ubuntu OS version required: {MIN_UBUNTU_VERSION}")

    npu_driver_version = subprocess.check_output(LINUX_NPU_COMMAND, shell=True, stderr=subprocess.STDOUT, universal_newlines=True).rstrip().split("\n")
    kernel_version = subprocess.check_output(LINUX_KERNEL_COMMAND, shell=True, stderr=subprocess.STDOUT, universal_newlines=True).rstrip()

    npu_driver_version = re.findall(r"\d\.\d\.\d", npu_driver_version[0])[0]
    linux_kernel_version = re.findall(r"\d\.\d", kernel_version)[0]

    return npu_driver_version, linux_kernel_version

def get_npu_info(core, os_type, npu_arch, npu_devid_selection):
    try:
        if os_type == "windows":
            return get_windows_npu_info(npu_arch, npu_devid_selection), None
        elif os_type == "linux":
            return get_linux_npu_info()
        else:
            raise ValueError(f"Unsupported OS type {os_type}")
    except Exception as e:
        print(f"Error getting NPU info: {e}")
        return None, None

if "ultra" in cpu_type.lower(): # bypass this test for RVP
    npu_arch = core.get_property('NPU', 'DEVICE_ARCHITECTURE')
    if npu_arch.upper() == "AUTO_DETECT":
        npu_arch = "4000"
    npu_devid_selection = mode_config['npu_devid_selection']
    npu_driver_version, linux_kernel_version = get_npu_info(core, os_type, npu_arch, npu_devid_selection)

core = Core()
cpu_type = core.get_property('CPU','full_device_name'.upper())
os_type = platform.system().lower()
npu_driver_version = None
npu_arch = None
npu_devid_selection= None
linux_kernel_version = None

# Constants
MIN_UBUNTU_VERSION = [22, 4]
LINUX_NPU_COMMAND = "dpkg -l | grep NPU  | awk '{print $3}'"
LINUX_KERNEL_COMMAND = "uname -r"
WINDOWS_NPU_COMMAND_TEMPLATE = "get-WmiObject Win32_PnPSignedDriver | Where-Object {{$_.DeviceID -like '{npu_devid}*'}} | Select-Object -ExpandProperty DriverVersion"

def get_windows_npu_info(npu_arch, npu_devid_selection):
    npu_devid = npu_devid_selection['windows'][npu_arch]
    command = WINDOWS_NPU_COMMAND_TEMPLATE.format(npu_devid=npu_devid)
    return subprocess.check_output(['powershell.exe', command], shell=True, universal_newlines=True).rstrip().split(".")

def get_linux_npu_info():
    linux_distro = distro.id().lower()
    os_version = list(map(int, distro.version().split('.')))
    
    if linux_distro != "ubuntu" or os_version < MIN_UBUNTU_VERSION:
        raise ValueError(f"Unsupported Linux Distro: {linux_distro} and OS Version: {os_version}; Minimum Ubuntu OS version required: {MIN_UBUNTU_VERSION}")

    npu_driver_version = subprocess.check_output(LINUX_NPU_COMMAND, shell=True, stderr=subprocess.STDOUT, universal_newlines=True).rstrip().split("\n")
    kernel_version = subprocess.check_output(LINUX_KERNEL_COMMAND, shell=True, stderr=subprocess.STDOUT, universal_newlines=True).rstrip()

    npu_driver_version = re.findall(r"\d\.\d\.\d", npu_driver_version[0])[0]
    linux_kernel_version = re.findall(r"\d\.\d", kernel_version)[0]

    return npu_driver_version, linux_kernel_version

def get_npu_info(core, os_type, npu_arch, npu_devid_selection):
    try:
        if os_type == "windows":
            return get_windows_npu_info(npu_arch, npu_devid_selection), None
        elif os_type == "linux":
            return get_linux_npu_info()
        else:
            raise ValueError(f"Unsupported OS type {os_type}")
    except Exception as e:
        print(f"Error getting NPU info: {e}")
        return None, None

if "ultra" in cpu_type.lower(): # bypass this test for RVP
    npu_arch = core.get_property('NPU', 'DEVICE_ARCHITECTURE')
    npu_devid_selection = mode_config['npu_devid_selection']
    npu_driver_version, linux_kernel_version = get_npu_info(core, os_type, npu_arch, npu_devid_selection)

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
                if model_name in revision_config:
                    revision = revision_config[model_name]['linux'][linux_kernel_version][npu_driver_version]
                    if revision and "main" not in revision:
                        revision += "-" + str(npu_arch)     
                else:
                    revision = revision_config['default']                 
        except KeyError:
            raise ValueError(f"Configuration mismatch! {os_type} & npu driver : {npu_driver_version} versions")
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
        
        if repo_id == "gblong1/sd-1.5-lcm-openvino":
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
    print("Downloading gblong1/sd-1.5-square-quantized Models")
    repo_id = "gblong1/sd-1.5-square-quantized"
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
    print("Downloading gblong1/sd-1.5-lcm-openvino")
    repo_id = "gblong1/sd-1.5-lcm-openvino"
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

def show_menu():
    """
    Display the menu options for downloading models.
    """
    print("=========Choose SD models to download=========")
    print("1 - SD-1.5 Square (512x512)")
    print("2 - SD-1.5 Portrait")
    print("3 - SD-1.5 Landscape")
    print("4 - SD-1.5 Inpainting (512x512 output image)")
    print("5 - SD-1.5 Controlnet-Openpose")
    print("6 - SD-1.5 Controlnet-CannyEdge")
    print("7 - SD-1.5 Controlnet-Scribble")
    print("8 - SD-1.5 LCM")
    print("9 - SD-1.5 Controlnet-ReferenceOnly")
    # print("10 - SD-2.1 Square (768x768)")
    print("12 - All the above models")
    print("0 - Exit SD Model setup")

def main():
    while True:
        show_menu()
        choice = input("Enter the number for the model you want to download: ")

        if choice == "0":
            print("Exiting SD Model setup...")
            break
        elif choice == "12":
            dl_all()
            print("Completed downloading all models. Exiting SD Model setup...")
            break
        else:
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
                # "10": dl_sd_21_square,
            }
            func = download_functions.get(choice)
            if func:
                func()
            else:
                print("Invalid choice")

if __name__ == "__main__":
    main()