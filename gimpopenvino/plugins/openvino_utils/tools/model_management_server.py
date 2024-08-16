import os
import json
import sys
import socket
import ast
import traceback
import logging as log
from pathlib import Path
import psutil
import threading

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")])
sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","openvino_utils","tools")])
from tools_utils import get_weight_path

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65434  # Port to listen on (stable_diffusion_ov_server uses port 65432 &  65433)

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


#TODO: This class should be in a separate utils file, so that it can be called from top-level model_setup.py
from openvino.runtime import Core 
from huggingface_hub import snapshot_download, HfFileSystem, hf_hub_url
import concurrent.futures
import platform
import shutil
import io
import requests
access_token = None

def compile_and_export_model(core, model_path, output_path, device='NPU', config=None):
    """
    Compile the model and export it to the specified path.
    """
    print("compile_and_export_model: model_path = ", model_path)
    model = core.compile_model(model_path, device, config=config)
    with io.BytesIO() as model_blob:
        model.export_model(model_blob)
        print("exporting model_blob to ", output_path)
        with open(output_path, 'wb') as f:
            f.write(model_blob.getvalue())
            
    print("compile_and_export_model: model_path = ", model_path, " done!")


def download_file_with_progress(url, local_filename, callback):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    
    percent_complete_last = -1.0;
    with open(local_filename, 'wb') as file:
        for data in response.iter_content(chunk_size=4096):
            file.write(data)
            downloaded_size += len(data)
            percent_complete = (downloaded_size / total_size) * 100
            
            if percent_complete - percent_complete_last > 5:
               percent_complete_last = percent_complete
               print(percent_complete,  "%")
               if callback:
                  callback(percent_complete)
               
            
            
class OpenVINOModelInstaller:
    def __init__(self):
        self._core = Core()
        self._os_type = platform.system().lower()
        self._available_devices = self._core.get_available_devices()
        self._npu_arch = None
        self._weight_path = get_weight_path() 
        self._install_location = os.path.join(self._weight_path, "stable-diffusion-ov")
        self._npu_arch = None
        if 'NPU' in self._available_devices:
            self._npu_arch = "3720" if "3720" in self._core.get_property('NPU', 'DEVICE_ARCHITECTURE') else "4000"
            
        self.hf_fs = HfFileSystem()
        
        
    def _generate_file_list_from_hf_repo_path(self, repo_id_path, file_list):
        repo_list = self.hf_fs.ls(repo_id_path, detail=True)
        #print("repo_list  = ", repo_list)
        for item in repo_list:
            item_name: str = item.get("name")
            item_type: str = item.get("type")
            if item_type == "directory":
                self._generate_file_list_from_hf_repo_path(item_name, file_list)
            else:
                if( item_type == "file" ):
                    file_list.append(item)
                    

    
    def _download_hf_repo(self, repo_id):
        file_list = []
        self._generate_file_list_from_hf_repo_path(repo_id, file_list)
        
        download_list = []
        total_file_list_size = 0
        for file in file_list:
            file_name: str = file.get("name")
            file_size: int = file.get("size")
            file_checksum: int = file.get("sha256")
            total_file_list_size += file_size
            #print(file_name)
            relative_path = os.path.relpath(file_name, repo_id)
            subfolder = os.path.dirname(relative_path).replace("\\", "/")
            relative_filename = os.path.basename(relative_path)
            url = hf_hub_url(
                    repo_id=repo_id, subfolder=subfolder, filename=relative_filename
                )
            download_list_item = {"filename": relative_path, "subfolder": subfolder, "size": file_size, "sha256": file_checksum, "url": url }
            download_list.append( download_list_item ) 
            print(download_list_item)
            
        print("total_file_list_size = ", total_file_list_size)
        
        download_folder = 'hf_download_folder'
        if os.path.isdir(download_folder):
            shutil.rmtree(download_folder)
            
        os.makedirs(download_folder)
        
        #okay, let's download the files one by one.
        for download_list_item in download_list:
           local_filename = os.path.join(download_folder, download_list_item["filename"])
           
           # create the subfolder (it may already exist, which is ok)
           subfolder=os.path.join(download_folder, download_list_item["subfolder"])
           os.makedirs(subfolder,  exist_ok=True)
           
           print("Downloading", download_list_item["url"], " to ", local_filename)
           download_file_with_progress(download_list_item["url"], local_filename, None)
           
        return download_folder
            
    def _download_quantized_models(self, repo_id, model_fp16, model_int8):
        download_flag = True
        SD_path_FP16 = os.path.join(self._install_location, model_fp16)
        SD_path_INT8 = os.path.join(self._install_location, model_int8)
        
        os.makedirs(SD_path_FP16,  exist_ok=True)

        if os.path.isdir(SD_path_FP16):
            choice = "Y"
            if choice == "Y" or choice == "y":
                shutil.rmtree(SD_path_FP16)
            else:
                download_flag = False
                print(f"{repo_id} download skipped")
                return download_flag
                
        if  download_flag:
            retries_left = 5
            while retries_left > 0:
                try:  
                    #download_folder = snapshot_download(repo_id=repo_id, token=access_token)
                    download_folder = self._download_hf_repo(repo_id)
                    break
                except Exception as e:
                    print("Error retry:" + str(e))
                    retries_left -= 1
 
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
            
            #delete_folder=os.path.join(download_folder, "..", "..", "..")
            #shutil.rmtree(delete_folder, ignore_errors=True)

                
        return download_flag
        
    def _download_model(self, repo_id, model_1, model_2):
        download_flag = True
        
        install_location=self._install_location
        
        if "sd-2.1" in repo_id:
            sd_model_1 = os.path.join(install_location, "stable-diffusion-2.1", model_1)    
        else:        
            sd_model_1 = os.path.join(install_location, "stable-diffusion-1.5", model_1)

        if os.path.isdir(sd_model_1):
            #choice = input(f"{repo_id} model folder exist. Do you wish to re-download this model? Enter Y/N: ")
            choice = "Y"
            if choice == "Y" or choice == "y":
                shutil.rmtree(sd_model_1)
            else:
                download_flag = False
                print(f"{repo_id} download skipped")
                return download_flag
                               
        if  download_flag:
            retries_left = 5
            while retries_left > 0:
                try:  
                    #download_folder = snapshot_download(repo_id=repo_id, token=access_token)
                    download_folder = self._download_hf_repo(repo_id)
                    break
                except Exception as e:
                    print("Error retry:" + str(e))
                    retries_left -= 1
            
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

            #delete_folder=os.path.join(download_folder, "../../..")
            #shutil.rmtree(delete_folder, ignore_errors=True)
    
        return download_flag

    def install_model(self, model_id):
        print("install_model: model_id=", model_id)
        if( model_id == "sd_15_square"):
            self.dl_sd_15_square()
        elif ( model_id == "sd_15_LCM"):
            self.dl_sd_15_LCM()
        else:
            print("Warning! unknown model_id=", model_id)
        
        print("install_model: model_id=", model_id, " done!")

    def dl_sd_15_square(self):
        print("Downloading Intel/sd-1.5-square-quantized Models")
        repo_id = "Intel/sd-1.5-square-quantized"
        model_fp16 = os.path.join("stable-diffusion-1.5", "square")
        model_int8 = os.path.join("stable-diffusion-1.5", "square_int8")
        install_location=self._install_location
        core = self._core
        npu_arch = self._npu_arch
        
        print("okay, actually downloading..")
        compile_models = self._download_quantized_models(repo_id, model_fp16, model_int8)
        #compile_models = True
        
        if npu_arch is not None:
            if not compile_models:
                #user_input = input("Do you want to reconfigure models for NPU? Enter Y/N: ").strip().lower()
                user_input="y"
                if user_input == "y":
                    compile_models = True
        
            if compile_models:
                text_future = None
                unet_int8_future = None
                unet_future = None
                vae_de_future = None        
                vae_en_future = None        
                
                if npu_arch == "3720":
                    # larger model should go first to avoid multiple checking when the smaller models loaded / compiled first
                    models_to_compile = [ "unet_int8", "text_encoder"]
                    shared_models = ["text_encoder.blob"]
                    sd15_futures = {
                        "text_encoder" : text_future,
                        "unet_int8" : unet_int8_future,
                    }
                else:
                    # also modified the model order for less checking in the future object when it gets result
                    models_to_compile = [ "unet_int8", "unet_bs1", "text_encoder", "vae_encoder" , "vae_decoder" ]
                    shared_models = ["text_encoder.blob", "vae_encoder.blob", "vae_decoder.blob"]
                    sd15_futures = {
                        "text_encoder" : text_future,
                        "unet_bs1" : unet_future,
                        "unet_int8" : unet_int8_future,
                        "vae_encoder" : vae_en_future,
                        "vae_decoder" : vae_de_future
                    }
        
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                        for model_name in models_to_compile:
                            model_path_fp16 = os.path.join(install_location, model_fp16, model_name + ".xml")
                            output_path_fp16 = os.path.join(install_location, model_fp16, model_name + ".blob")
                            if "unet_int8" in model_name:
                                model_path_int8 = os.path.join(install_location, model_int8, model_name + ".xml")
                                output_path_int8 = os.path.join(install_location, model_int8, model_name + ".blob")
                                print(f"Creating NPU model for {model_name}")
                                sd15_futures[model_name] = executor.submit(compile_and_export_model, core, model_path_int8, output_path_int8)
                            else:
                                print(f"Creating NPU model for {model_name}")
                                sd15_futures[model_name] = executor.submit(compile_and_export_model, core, model_path_fp16, output_path_fp16)
                     
                    if npu_arch == "3720":                  
                        sd15_futures["unet_int8"].result()
                        sd15_futures["text_encoder"].result()
                    else:
                        sd15_futures["unet_int8"].result()
                        sd15_futures["unet_bs1"].result()
                        sd15_futures["vae_decoder"].result()
                        sd15_futures["vae_encoder"].result()
                        sd15_futures["text_encoder"].result()
                except Exception as e:
                    print("Compilation failed. Exception: ")
                    print(e)
                    return 
            
                

                      
                # Copy shared models to INT8 directory
                print("copying shared models... install_location=", install_location)
                for blob_name in shared_models:
                    print("shutil.copy(", os.path.join(install_location, model_fp16, blob_name),",",os.path.join(install_location, model_int8, blob_name))
                    shutil.copy(
                        os.path.join(install_location, model_fp16, blob_name),
                        os.path.join(install_location, model_int8, blob_name)
                    )
                #:::::::::::::: START REMOVE ME ::::::::::::::
                # Temporary workaround to force the config for Lunar Lake - 
                # REMOVE ME before publishing to external open source.    
                config_data = { 	"power modes supported": "yes", 	
                                        "best performance" : ["GPU","GPU","GPU","GPU"],
                                                "balanced" : ["NPU","NPU","GPU","GPU"],
                                   "best power efficiency" : ["NPU","NPU","NPU","NPU"]
                                }
                # Specify the file name
                file_name = "config.json"

                # Write the data to a JSON file
                with open(os.path.join(install_location, model_fp16, file_name), 'w') as json_file:
                    json.dump(config_data, json_file, indent=4)
                # Write the data to a JSON file
                with open(os.path.join(install_location, model_int8, file_name), 'w') as json_file:
                    json.dump(config_data, json_file, indent=4)
                #:::::::::::::: END REMOVE ME ::::::::::::::
                
    
    def dl_sd_15_LCM(self):
        print("Downloading Intel/sd-1.5-lcm-openvino")
        repo_id = "Intel/sd-1.5-lcm-openvino"
        model_1 = "square_lcm"
        model_2 = None
        compile_models = self._download_model(repo_id, model_1, model_2)
        install_location=self._install_location
        core = self._core
        npu_arch = self._npu_arch

        
        if npu_arch is not None:
            if not compile_models:
                user_input = input("Do you want to reconfigure models for NPU? Enter Y/N: ").strip().lower()
                if user_input == "y":
                    compile_models = True
        
            if compile_models:  
                text_future = None
                unet_future = None
                vae_de_future = None
                
                if npu_arch == "3720":
                    models_to_compile = [ "unet", "text_encoder"]
                    sd15_futures = {
                        "text_encoder" : text_future,
                        "unet" : unet_future,
                    }
                else:
                    models_to_compile = [ "unet" , "vae_decoder", "text_encoder" ]
                    sd15_futures = {
                        "text_encoder" : text_future,
                        "unet" : unet_future,
                        "vae_decoder" : vae_de_future
                    }
        
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                        for model_name in models_to_compile:
                            model_path = os.path.join(install_location, "stable-diffusion-1.5", model_1, model_name + ".xml")
                            output_path = os.path.join(install_location,"stable-diffusion-1.5", model_1, model_name + ".blob")
                            print(f"Creating NPU model for {model_name}")
                            sd15_futures[model_name] = executor.submit(compile_and_export_model, core, model_path, output_path)
                     
                    if npu_arch == "3720":                  
                        sd15_futures["text_encoder"].result()
                        sd15_futures["unet"].result()
                    else:
                        sd15_futures["text_encoder"].result()
                        sd15_futures["unet"].result()
                        sd15_futures["vae_decoder"].result()
                except:
                    print("Compilation failed.")    






def run():
    print("model management server starting...")
    weight_path = get_weight_path()
    
    #Move to a temporary working directory in a known place. 
    # This is where we'll be downloading stuff to, etc.
    tmp_working_dir=os.path.join(weight_path, '..', 'mms_tmp')
    
    #if this dir doesn't exist, create it.
    if not os.path.isdir(tmp_working_dir):
        os.mkdir(tmp_working_dir)
    
    # go there.    
    os.chdir(tmp_working_dir)
    
    ov_model_installer = OpenVINOModelInstaller() 
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        while True:
            print("awaiting new connection..")
            conn, addr = s.accept()
            print("new connection established..")
            with conn:
                while True:
                    print("Model Management Server Waiting..")
                    data = conn.recv(1024)
                    
                    if not data:
                        break
                    
                    if data.decode() == "kill":
                        os._exit(0)
                    if data.decode() == "ping":
                        conn.sendall(data)
                        continue
                    if data.decode() == "install_model":
                       print("Model Management Server: install_model cmd received. Getting model name..")
                       #send ack
                       conn.sendall(data)

                       #get model id.
                       #TODO: Need a timeout here.
                       model_id = conn.recv(1024).decode()
                       
                       print("Model Management Server: model_id=", model_id)
                       
                       #send ack
                       conn.sendall(data)
                       
                       #todo: run on another thread, so that we can service other requests while this one is downloading.
                       ov_model_installer.install_model(model_id)
                       
                       continue
                       
                    print("Warning! Unsupported command sent: ", data.decode())
                    
                    
                        
    print("model management server exiting...")
    


def start():

    run_thread = threading.Thread(target=run, args=())
    run_thread.start()
    
    gimp_proc = None
    for proc in psutil.process_iter():
        if "gimp-2.99" in proc.name():
            gimp_proc = proc
            break
            
    if gimp_proc:
        psutil.wait_procs([proc])
        print("model management server exiting..!")
        os._exit(0)

    run_thread.join()
    
if __name__ == "__main__":
   start()
