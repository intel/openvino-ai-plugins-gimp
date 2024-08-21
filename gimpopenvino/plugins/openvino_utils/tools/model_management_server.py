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

#TODO: Put this in a standalone py, or json config, etc. Someplace outside of model_management_server.py.
g_supported_models = [
    {
        "name": "Stable Diffusion 1.5 (Square)",
        "description": "A short description of Stable Diffusion 1.5.",
        "id": "sd_15_square",
        "install_subdir": os.path.join("stable-diffusion-1.5", "square")
    },
    {
        "name": "Stable Diffusion 1.5 LCM",
        "description": "A short description of Stable Diffusion 1.5 LCM.",
        "id": "sd_15_LCM",
        "install_subdir": os.path.join("stable-diffusion-1.5", "square_lcm")
    },
    {
        "name": "Stable Diffusion 1.5 (Portrait)",
        "description": "A short description of Stable Diffusion 1.5.",
        "id": "sd_15_portrait",
        "install_subdir": os.path.join("stable-diffusion-1.5", "portrait")
    },
    {
        "name": "Stable Diffusion 1.5 (Landscape)",
        "description": "A short description of Stable Diffusion 1.5.",
        "id": "sd_15_landscape",
        "install_subdir": os.path.join("stable-diffusion-1.5", "landscape")
    },
    {
        "name": "Stable Diffusion 1.5 (Inpainting)",
        "description": "A short description of Stable Diffusion 1.5.",
        "id": "sd_15_inpainting",
        "install_subdir": os.path.join("stable-diffusion-1.5", "inpainting")
    },
    {
        "name": "Stable Diffusion 1.5 (Controlnet OpenPose)",
        "description": "A short description of Stable Diffusion 1.5.",
        "id": "sd_15_openpose",
        "install_subdir": os.path.join(".", "controlnet-openpose")
    },
    {
        "name": "Stable Diffusion 1.5 (Controlnet Canny)",
        "description": "A short description of Stable Diffusion 1.5.",
        "id": "sd_15_canny",
        "install_subdir": os.path.join(".", "controlnet-canny")
    },
    {
        "name": "Stable Diffusion 1.5 (Controlnet Scribble)",
        "description": "A short description of Stable Diffusion 1.5.",
        "id": "sd_15_scribble",
        "install_subdir": os.path.join(".", "controlnet-scribble")
    },
    {
        "name": "Stable Diffusion 1.5 (Controlnet Reference-Only)",
        "description": "A short description of Stable Diffusion 1.5.",
        "id": "sd_15_Referenceonly",
        "install_subdir": os.path.join(".", "controlnet-referenceonly")
    },

    {
        "name": "Test Model 1",
        "description": "This is just a test entry that doesn't actually download anything.",
        "id": "test1", "install_subdir": os.path.join("test")
    },

    {
        "name": "Test Model 2",
        "description": "This is just a test entry that doesn't actually download anything.",
        "id": "test2", "install_subdir": os.path.join("test")
    },
]

#TODO: This class should be in a separate utils file, so that it can be called from top-level model_setup.py
from openvino.runtime import Core
from huggingface_hub import snapshot_download, HfFileSystem, hf_hub_url
import concurrent.futures
import platform
import shutil
import io
import requests
import queue
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


def download_file_with_progress(url, local_filename, callback, total_bytes_downloaded, total_file_list_size):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0

    percent_complete_last = -1.0;
    with open(local_filename, 'wb') as file:
        for data in response.iter_content(chunk_size=4096):
            file.write(data)
            downloaded_size += len(data)
            total_bytes_downloaded += len(data)
            percent_complete = (downloaded_size / total_size) * 100

            if percent_complete - percent_complete_last > 1:
               percent_complete_last = percent_complete
               #print(percent_complete,  "%")
               if callback:
                  callback(total_bytes_downloaded, total_file_list_size)


    return downloaded_size


class OpenVINOModelInstaller:
    def __init__(self):
        print("OpenVINOModelInstaller..")
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
        self.model_install_status = {}
        self.install_queue = queue.Queue()
        self.install_lock = threading.Condition()


    def get_all_model_details(self):
        model_details = []
        for model_detail in g_supported_models:
            model_check_path = os.path.join(self._install_location, model_detail["install_subdir"], )

            model_detail_entry = model_detail.copy()

            if model_detail["id"] in  self.model_install_status:
                install_status = "installing"
            elif not os.path.isdir(model_check_path):
                install_status = "not_installed"
            else:
                install_status = "installed"

            model_detail_entry['install_status'] = install_status

            model_details.append(model_detail_entry)

        return model_details

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



    def _download_hf_repo(self, repo_id, model_id):
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


        def bytes_downloaded_callback(total_bytes_downloaded, total_bytes_to_download):


           total_bytes_to_download_gb = total_bytes_to_download / 1073741824.0
           total_bytes_to_download_gb = f"{total_bytes_to_download_gb:.2f}"
           total_bytes_downloadeded_gb = total_bytes_downloaded / 1073741824.0
           total_bytes_downloadeded_gb = f"{total_bytes_downloadeded_gb:.2f}"
           status = "Downloading... (" + total_bytes_downloadeded_gb + " / " + total_bytes_to_download_gb + ") GiB"
           if total_bytes_to_download > 0:
               self.model_install_status[model_id]["status"] = status
               self.model_install_status[model_id]["percent"] = (total_bytes_downloaded / total_bytes_to_download) * 100.0


        total_bytes_downloaded = 0
        #okay, let's download the files one by one.
        for download_list_item in download_list:
           local_filename = os.path.join(download_folder, download_list_item["filename"])

           # create the subfolder (it may already exist, which is ok)
           subfolder=os.path.join(download_folder, download_list_item["subfolder"])
           os.makedirs(subfolder,  exist_ok=True)

           print("Downloading", download_list_item["url"], " to ", local_filename)


           downloaded_size = download_file_with_progress(download_list_item["url"], local_filename, bytes_downloaded_callback, total_bytes_downloaded, total_file_list_size)

           total_bytes_downloaded += downloaded_size

        return download_folder

    def _download_quantized_models(self, repo_id, model_fp16, model_int8, model_id):
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
            download_success = False
            while retries_left > 0:
                try:
                    #download_folder = snapshot_download(repo_id=repo_id, token=access_token)
                    download_folder = self._download_hf_repo(repo_id, model_id)
                    download_success = True
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


            if download_success is True:
                shutil.rmtree(download_folder, ignore_errors=True)


        return download_flag

    def _download_model(self, repo_id, model_1, model_2, model_id):
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
            download_success = False
            while retries_left > 0:
                try:
                    download_folder = self._download_hf_repo(repo_id, model_id)
                    download_success = True
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

            if download_success is True:
                shutil.rmtree(download_folder, ignore_errors=True)

        return download_flag


    def install_test(self, model_id):

        import time

        states = [ "Downloading...", "Prepping NPU Models.."]

        for state in states:
            percent_complete = 0.0
            self.model_install_status[model_id]["status"] = state
            self.model_install_status[model_id]["percent"] = percent_complete

            last_perc_complete_printed = 0.0

            while percent_complete < 100:
                time.sleep(0.1)
                #percent_complete += 0.2
                percent_complete += 1

                self.model_install_status[model_id]["percent"] = percent_complete

                if( percent_complete - last_perc_complete_printed > 10 ):
                    print("install_test %: ", percent_complete)
                    last_perc_complete_printed = percent_complete


    def install_model(self, model_id):
        print("install_model: model_id=", model_id)

        # set the status to 'Queued' this is what will display in the UI
        # until it's this model's turn to get installed.
        self.model_install_status[model_id] = {"status": "Queued", "percent": 0.0}

        # Put this thread into the quueue
        self.install_queue.put(threading.current_thread())

        # acquire the lock
        with self.install_lock:

            # check to see if it's our turn
            while self.install_queue.queue[0] != threading.current_thread():
                # it's not our turn.. go back to sleep
                lock.wait()

            # Dequeue the thread now that it's our turn
            self.install_queue.get()

            if( model_id == "sd_15_square"):
                self.dl_sd_15_square(model_id)
            elif ( model_id == "sd_15_LCM"):
                self.dl_sd_15_LCM(model_id)
            elif ( model_id == "sd_15_portrait"):
                self.dl_sd_15_portrait(model_id)
            elif ( model_id == "sd_15_landscape"):
                self.dl_sd_15_landscape(model_id)
            elif ( model_id == "sd_15_inpainting"):
                self.dl_sd_15_inpainting(model_id)
            elif ( model_id == "sd_15_openpose"):
                self.dl_sd_15_openpose(model_id)
            elif ( model_id == "sd_15_canny"):
                self.dl_sd_15_canny(model_id)
            elif ( model_id == "sd_15_scribble"):
                self.dl_sd_15_scribble(model_id)
            elif ( model_id == "sd_15_Referenceonly"):
                self.dl_sd_15_Referenceonly(model_id)
            elif (model_id == "test1"):
                self.install_test(model_id)
            elif (model_id == "test2"):
                self.install_test(model_id)
            else:
                print("Warning! unknown model_id=", model_id)

            # Notify the next thread in the queue
            self.install_lock.notify_all()

        self.model_install_status.pop(model_id)

        print("install_model: model_id=", model_id, " done!")

    def dl_sd_15_portrait(self, model_id):
        print("Downloading Intel/sd-1.5-portrait-quantized Models")
        repo_id = "Intel/sd-1.5-portrait-quantized"
        model_1 = "portrait"
        model_2 = "portrait_512x768"
        self._download_model(repo_id, model_1, model_2, model_id)

    def dl_sd_15_landscape(self, model_id):
        print("Downloading Intel/sd-1.5-landscape-quantized Models")
        repo_id = "Intel/sd-1.5-landscape-quantized"
        model_1 = "landscape"
        model_2 = "landscape_768x512"
        self._download_model(repo_id, model_1, model_2, model_id)

    def dl_sd_15_inpainting(self, model_id):
        print("Downloading Intel/sd-1.5-inpainting-quantized Models")
        repo_id = "Intel/sd-1.5-inpainting-quantized"
        model_fp16 = os.path.join("stable-diffusion-1.5", "inpainting")
        model_int8 = os.path.join("stable-diffusion-1.5", "inpainting_int8")
        self._download_quantized_models(repo_id, model_fp16, model_int8, model_id)

    def dl_sd_15_openpose(self, model_id):
        print("Downloading Intel/sd-1.5-controlnet-openpose-quantized Models")
        repo_id="Intel/sd-1.5-controlnet-openpose-quantized"
        model_fp16 = "controlnet-openpose"
        model_int8 = "controlnet-openpose-int8"
        self._download_quantized_models(repo_id, model_fp16,model_int8, model_id)

    def dl_sd_15_canny(self, model_id):
        print("Downloading Intel/sd-1.5-controlnet-canny-quantized Models")
        repo_id = "Intel/sd-1.5-controlnet-canny-quantized"
        model_fp16 = "controlnet-canny"
        model_int8 = "controlnet-canny-int8"
        self._download_quantized_models(repo_id, model_fp16, model_int8, model_id)

    def dl_sd_15_scribble(self, model_id):
        print("Downloading Intel/sd-1.5-controlnet-scribble-quantized Models")
        repo_id = "Intel/sd-1.5-controlnet-scribble-quantized"
        model_fp16 = "controlnet-scribble"
        model_int8 = "controlnet-scribble-int8"
        self._download_quantized_models(repo_id, model_fp16, model_int8, model_id)

    def dl_sd_15_Referenceonly(self, model_id):
        print("Downloading Intel/sd-reference-only")
        repo_id = "Intel/sd-reference-only"
        model_fp16 = "controlnet-referenceonly"
        model_int8 = None
        self._download_model(repo_id, model_fp16, model_int8, model_id)

    def dl_sd_15_square(self, model_id):
        print("Downloading Intel/sd-1.5-square-quantized Models")
        repo_id = "Intel/sd-1.5-square-quantized"
        model_fp16 = os.path.join("stable-diffusion-1.5", "square")
        model_int8 = os.path.join("stable-diffusion-1.5", "square_int8")
        install_location=self._install_location
        core = self._core
        npu_arch = self._npu_arch

        compile_models = self._download_quantized_models(repo_id, model_fp16, model_int8, model_id)
        #compile_models = True

        if npu_arch is not None:
            if not compile_models:
                #user_input = input("Do you want to reconfigure models for NPU? Enter Y/N: ").strip().lower()
                user_input="y"
                if user_input == "y":
                    compile_models = True

            if compile_models:
                self.model_install_status[model_id]["status"] = "Compiling models for NPU..."
                self.model_install_status[model_id]["percent"] = 0.0
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


                            num_futures = len(sd15_futures)
                            perc_increment = 100.0 / num_futures

                            self.model_install_status[model_id]["percent"] = 0.0
                            for model_name, model_future in sd15_futures.items():
                                model_future.result()
                                self.model_install_status[model_id]["percent"] += perc_increment

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


    def dl_sd_15_LCM(self, model_id):
        print("Downloading Intel/sd-1.5-lcm-openvino")
        repo_id = "Intel/sd-1.5-lcm-openvino"
        model_1 = "square_lcm"
        model_2 = None
        compile_models = self._download_model(repo_id, model_1, model_2, model_id)
        install_location=self._install_location
        core = self._core
        npu_arch = self._npu_arch


        if npu_arch is not None:
            if not compile_models:
                user_input = input("Do you want to reconfigure models for NPU? Enter Y/N: ").strip().lower()
                if user_input == "y":
                    compile_models = True

            if compile_models:
                self.model_install_status[model_id]["status"] = "Compiling models for NPU..."
                self.model_install_status[model_id]["percent"] = 0.0
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

                        num_futures = len(sd15_futures)
                        perc_increment = 100.0 / num_futures

                        self.model_install_status[model_id]["percent"] = 0.0
                        for model_name, model_future in sd15_futures.items():
                            model_future.result()
                            self.model_install_status[model_id]["percent"] += perc_increment
                except:
                    print("Compilation failed.")




def run_connection_routine(ov_model_installer, conn):
    with conn:
        while True:
            data = conn.recv(1024)

            if not data:
                break

            if data.decode() == "kill":
                os._exit(0)
            if data.decode() == "ping":
                conn.sendall(data)
                continue

            # request to get the details and state of all supported models
            if data.decode() == "get_all_model_details":
                model_details = ov_model_installer.get_all_model_details()

                #get number of models
                num_models = len(model_details)

                print("num_models = ", num_models)
                conn.sendall(bytes(str(num_models), 'utf-8'))
                #wait for ack
                data = conn.recv(1024)

                for i in range(0, num_models):
                    for detail in ["name", "description", "id", "install_status"]:
                        conn.sendall(bytes(model_details[i][detail], 'utf-8'))
                        #wait for ack
                        data = conn.recv(1024)

                continue

            # request to install a model.
            if data.decode() == "install_model":
               print("Model Management Server: install_model cmd received. Getting model name..")
               #send ack
               conn.sendall(data)

               #get model id.
               #TODO: Need a timeout here.
               model_id = conn.recv(1024).decode()

               print("Model Management Server: model_id=", model_id)

               if model_id not in ov_model_installer.model_install_status:

                   # add it to the dictionary here (instead of in ov_model_installer.install_model).
                   # This will guarantee that it is present in the dictionary before sending the ack,
                   #  and avoiding a potential race condition where the GIMP UI side asks for the status
                   #  before the thread spawns.
                   ov_model_installer.model_install_status[model_id] = {"status": "Preparing to install..", "percent": 0.0}

                   #Run the install on another thread. This allows the server to service other requests
                   # while the install is taking place.
                   install_thread = threading.Thread(target=ov_model_installer.install_model, args=(model_id,))
                   install_thread.start()
               else:
                   print(model_id, "is already currently installing!")

               #send ack
               conn.sendall(data)

               #ov_model_installer.install_model(model_id)

               continue

            # request to get the status of a model that is getting installed.
            if data.decode() == "install_status":

                # send ack
                conn.sendall(data)

                # make a copy of this so that the number of entries doesn't change while we're
                #  in this routine.
                model_install_status = ov_model_installer.model_install_status.copy()

                # Get the model-id that we are interested in.
                data = conn.recv(1024)
                model_id = data.decode()

                if model_id in model_install_status:
                    details = model_install_status[model_id]

                    status = details["status"]
                    perc = details["percent"]
                else:
                    # the model_id is not found in the installer map... set status to "done" / 100.0
                    # TODO: What about failure cases?
                    status = "done"
                    perc = 100.0

                # first, send the status
                conn.sendall(bytes(status, 'utf-8'))
                data = conn.recv(1024) # <- get ack

                # then, send the send the percent
                conn.sendall(bytes(str(perc), 'utf-8'))
                data = conn.recv(1024) # <- get ack

                continue

            print("Warning! Unsupported command sent: ", data.decode())

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

            #Run this connection on a dedicated thread. This allows multiple connections to be present at once.
            conn_thread = threading.Thread(target=run_connection_routine, args=(ov_model_installer, conn))
            conn_thread.start()
            #run_connection_routine(ov_model_installer, conn)




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
