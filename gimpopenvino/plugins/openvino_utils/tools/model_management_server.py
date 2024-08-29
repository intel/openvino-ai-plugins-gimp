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

#TODO: Put these in a standalone py, or json config, etc. Someplace outside of model_management_server.py.


# This dictionary is used to populate the drop-down model selection list.
# It's a map from model-id -> model_details.
# 'install_id' is the key used for the 'installable model map'
#  Note that two models in this map may have the same install_id -- this just
#   means a single 'Install' entry in the Model Manager UI will install both of them.
#  If 'install_id' is None, it means that we don't (yet) support installing this model
#   from the Model Manager UI (but we'll still populate the model selection drop-down
#   with this model, if we detect that it is available).
g_supported_model_map = {
    "sd_1.5_square":
    {
        "name": "Stable Diffusion 1.5 (Square 512x512)(FP16)",
        "install_id": "sd_15_square",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "square"]
    },

    "sd_1.5_square_int8":
    {
        "name": "Stable Diffusion 1.5 (Square 512x512)(INT8)",
        "install_id": "sd_15_square",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"]
    },

    "sd_1.5_square_lcm":
    {
        "name": "Stable Diffusion 1.5 LCM (Square 512x512)(INT8)",
        "install_id": "sd_15_LCM",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_lcm"],
    },

    "sd_3.0_square_int8":
    {
        "name": "Stable Diffusion 3.0 (Square 512x512)(INT8)",
        "install_id": None, # Set to None, so that model manager UI doesn't give option to install.
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-3.0", "square_int8"],
    },

    "sd_3.0_square_int4":
    {
        "name": "Stable Diffusion 3.0 (Square 512x512)(INT4)",
        "install_id": None, # Set to None, so that model manager UI doesn't give option to install.
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-3.0", "square_int4"],
    },

    "sd_1.5_portrait":
    {
        "name": "Stable Diffusion 1.5 (Portrait 360x640)(INT8)",
        "install_id": "sd_15_portrait",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait"]
    },

    "sd_1.5_portrait_512x768":
    {
        "name": "Stable Diffusion 1.5 (Portrait 512x768)(INT8)",
        "install_id": "sd_15_portrait",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait_512x768"],
    },

    "sd_1.5_landscape":
    {
        "name": "Stable Diffusion 1.5 (Landscape 640x360)(INT8)",
        "install_id": "sd_15_landscape",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape"],
    },

    "sd_1.5_landscape_768x512":
    {
        "name": "Stable Diffusion 1.5 (Landscape 768x512)(INT8)",
        "install_id": "sd_15_landscape",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape_768x512"],
    },

    "sd_1.5_inpainting":
    {
        "name": "Stable Diffusion 1.5 (Inpainting)(FP16)",
        "install_id": "sd_15_inpainting",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting"],
    },

    "sd_1.5_inpainting_int8":
    {
        "name": "Stable Diffusion 1.5 (Inpainting)(INT8)",
        "install_id": "sd_15_inpainting",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting_int8"],
    },

    "controlnet_openpose":
    {
        "name": "Stable Diffusion 1.5 (Controlnet OpenPose)(FP16)",
        "install_id": "sd_15_openpose",
        "install_subdir": ["stable-diffusion-ov", "controlnet-openpose"],
    },

    "controlnet_openpose_int8":
    {
        "name": "Stable Diffusion 1.5 (Controlnet OpenPose)(INT8)",
        "install_id": "sd_15_openpose",
        "install_subdir": ["stable-diffusion-ov", "controlnet-openpose-int8"],
    },

    "controlnet_canny":
    {
        "name": "Stable Diffusion 1.5 (Controlnet Canny)(FP16)",
        "install_id": "sd_15_canny",
        "install_subdir": ["stable-diffusion-ov", "controlnet-canny"],
    },

    "controlnet_canny_int8":
    {
        "name": "Stable Diffusion 1.5 (Controlnet Canny)(INT8)",
        "install_id": "sd_15_canny",
        "install_subdir": ["stable-diffusion-ov", "controlnet-canny-int8"],
    },

    "controlnet_scribble":
    {
        "name": "Stable Diffusion 1.5 (Controlnet Scribble)(FP16)",
        "install_id": "sd_15_scribble",
        "install_subdir": ["stable-diffusion-ov", "controlnet-scribble"],
    },

    "controlnet_scribble_int8":
    {
        "name": "Stable Diffusion 1.5 (Controlnet Scribble)(INT8)",
        "install_id": "sd_15_scribble",
        "install_subdir": ["stable-diffusion-ov", "controlnet-scribble-int8"],
    },

    "controlnet_referenceonly":
    {
        "name": "Stable Diffusion 1.5 (Controlnet Reference-Only)(FP16)",
        "install_id": "sd_15_Referenceonly",
        "install_subdir": ["stable-diffusion-ov", "controlnet-referenceonly"],
    },

    "test.1":
    {
        "name": "Test 1 Model",
        "install_id": "test1",
        "install_subdir": ["stable-diffusion-ov", "test1"],
    },

    "test.2":
    {
        "name": "Test 2 Model",
        "install_id": "test2",
        "install_subdir": ["stable-diffusion-ov", "test2"],
    },

}


# The thing used to populate the Model Manager UI (or model_setup.py console selections)
# Each model in g_supported_model_map define 'install_id',
#  which is the key to use.
# It's called 'base' model map, since it's meant to be
#  an initializer.. not something to use as-is. For example,
#  OpenVINOModelInstaller makes a copy of it, and actually adds
#  more details to each entry.
# name: The name/title to be displayed in the Model Manager UI
# repo_id: HF repo id -- used by download routine.
# download_exclude_filters: an array of glob patterns. This is used to exclude downloading
#                           unnecessary files.
g_installable_base_model_map = {
    "sd_15_square":
    {
        "name": "Stable Diffusion 1.5 (Square)",
        "repo_id": "Intel/sd-1.5-square-quantized",
        "download_exclude_filters": ["*.blob"],
    },

    "sd_15_LCM":
    {
        "name": "Stable Diffusion 1.5 LCM",
        "repo_id": "Intel/sd-1.5-lcm-openvino",
        "download_exclude_filters": [],
    },

    "sd_15_portrait":
    {
        "name": "Stable Diffusion 1.5 (Portrait)",
        "repo_id": "Intel/sd-1.5-portrait-quantized",
        "download_exclude_filters": [],
    },

    "sd_15_landscape":
    {
        "name": "Stable Diffusion 1.5 (Landscape)",
        "repo_id": "Intel/sd-1.5-landscape-quantized",
        "download_exclude_filters": [],
    },

    "sd_15_inpainting":
    {
        "name": "Stable Diffusion 1.5 (Inpainting)",
        "repo_id": "Intel/sd-1.5-inpainting-quantized",
        "download_exclude_filters": [],
    },

    "sd_15_openpose":
    {
        "name": "Stable Diffusion 1.5 (Controlnet OpenPose)",
        "repo_id": "Intel/sd-1.5-controlnet-openpose-quantized",
        "download_exclude_filters": [],
    },

    "sd_15_canny":
    {
        "name": "Stable Diffusion 1.5 (Controlnet Canny)",
        "repo_id": "Intel/sd-1.5-controlnet-canny-quantized",
        "download_exclude_filters": [],
    },

    "sd_15_scribble":
    {
        "name": "Stable Diffusion 1.5 (Controlnet Canny)",
        "repo_id": "Intel/sd-1.5-controlnet-scribble-quantized",
        "download_exclude_filters": [],
    },

    "sd_15_Referenceonly":
    {
        "name": "Stable Diffusion 1.5 (Controlnet Reference-Only)",
        "repo_id": "Intel/sd-reference-only",
        "download_exclude_filters": [],
    },

    "test1":
    {
        "name": "Test Model 1",
        "repo_id": None,
        "download_exclude_filters": [],
    },

    "test2":
    {
        "name": "Test Model 2",
        "repo_id": None,
        "download_exclude_filters": [],
    }
}

#TODO: This class should be in a separate utils file, so that it can be called from top-level model_setup.py
from openvino.runtime import Core
from huggingface_hub import snapshot_download, HfFileSystem, hf_hub_url
import concurrent.futures
import platform
import shutil
import io
import requests
import queue
import fnmatch
access_token = None

def does_filename_match_patterns(filename, patterns):
    for pattern in patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False

def is_subdirectory(child_path, parent_path):
    # Convert to absolute paths
    child_path = Path(child_path).resolve()
    parent_path = Path(parent_path).resolve()

    # Check if the parent path is a prefix of the child path
    return parent_path in child_path.parents

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
                  # if the callback returns True, the user cancelled. So just return right now.
                  if callback(total_bytes_downloaded, total_file_list_size):
                      return downloaded_size

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
        self.model_install_status_lock = threading.Lock()
        self.install_queue = queue.Queue()
        self.install_lock = threading.Condition()

        # make a copy of the base map
        self.installable_model_map = g_installable_base_model_map.copy()

        # first, add empty array as 'supported_model_ids' key (so that we can simply append in next looop)
        for install_details in self.installable_model_map.values():
            install_details["supported_model_ids"] = []

        # for each supported model, append id to the 'supported_model_ids' array.
        for supported_model_id, supported_model_details in g_supported_model_map.items():
            install_id =  supported_model_details["install_id"]

            # Note: This is not an error condition. It just means that we don't want the installation
            #  of this model to be controlled by the model management UI.
            if not install_id:
                continue

            if install_id not in self.installable_model_map:
                print("Unexpected error: install_id=", install_id, " not present in installable model map..")
                continue

            self.installable_model_map[install_id]["supported_model_ids"].append(supported_model_id)


    # Given a model_id, is it installed? This is a very simple check right now (check for existence of directory)
    #  but it likely needs to be transformed into something better (i.e. reading a json from the directly, cross-checking
    #  HF commit-id, etc.)
    def is_model_installed(self, model_id):

        if model_id not in g_supported_model_map:
            return False

        install_subdir = g_supported_model_map[model_id]["install_subdir"]

        model_check_path = os.path.join(self._weight_path, *install_subdir)

        if os.path.isdir(model_check_path):
            return True

        return False


    # This function returns 2 things:
    # 1. The list of (individual) models that are installed. Essentially, this is used to populate the model selection drop-down list.
    #    For each model, it will send:
    #       - the model id. e.g. "sd_1.5_square"
    #       - the 'full' model name. e.g. "Stable Diffusion 1.5 (Square 512x512)(FP16)"
    #
    # 2. The list of installable models. Essentially, this is used to populate rows for the model manager ui window.
    #    For each installable model, it will send:
    #       - the installable model id. e.g. "sd_15_square"
    #       - the full name. e.g. "Stable Diffusion 1.5 (Square)"
    #       - the state of installation ('installed', 'not_installed', or 'installing')
    def get_all_model_details(self):

        # okay, first we need to create a list of models that we detect as being installed.
        installed_models = []
        for model_id, model_details in g_supported_model_map.items():
            if self.is_model_installed(model_id):
                installed_models.append( {"id": model_id, "name": model_details["name"]} )

        installable_model_details = []
        for install_id, install_details in self.installable_model_map.items():

            # If all models in supported_model_ids are installed, then we give
            #  an overall status of 'installed'.
            # If the install_id is currently in the model_install_status map,
            #  then we set overall status to 'installing'.
            # If any of the models are *not* installed, then we give an overall status
            #  of 'not_installed'
            if install_id in self.model_install_status:
                install_status = "installing"
            else:
                all_are_installed = True

                for model_id in install_details["supported_model_ids"]:
                    if self.is_model_installed(model_id) is False:
                        all_are_installed = False
                        break

                if all_are_installed:
                    install_status = "installed"
                else:
                    install_status = "not_installed"

                #TODO: this will get refactored.
                model_detail_entry = {}
                model_detail_entry["name"] = install_details["name"]
                model_detail_entry["description"] = "Some unused description"
                model_detail_entry["id"] = install_id
                model_detail_entry['install_status'] = install_status

                installable_model_details.append(model_detail_entry)

        return installed_models, installable_model_details

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



    def _download_hf_repo(self, repo_id, model_id, download_folder, exclude_filters = None):

        retries_left = 5
        while retries_left > 0:
            try:
                file_list = []
                if os.path.isdir(download_folder):
                    shutil.rmtree(download_folder)

                os.makedirs(download_folder)

                self.model_install_status[model_id]["status"] = "Downloading..."
                self.model_install_status[model_id]["percent"] = 0.0

                self._generate_file_list_from_hf_repo_path(repo_id, file_list)
                download_list = []
                total_file_list_size = 0
                for file in file_list:
                    file_name: str = file.get("name")
                    file_size: int = file.get("size")
                    file_checksum: int = file.get("sha256")

                    #print(file_name)
                    relative_path = os.path.relpath(file_name, repo_id)

                    if exclude_filters:
                        if does_filename_match_patterns(relative_path, exclude_filters):
                            print(relative_path, ": Skipped due to exclude filters")
                            continue

                    total_file_list_size += file_size

                    subfolder = os.path.dirname(relative_path).replace("\\", "/")
                    relative_filename = os.path.basename(relative_path)
                    url = hf_hub_url(
                            repo_id=repo_id, subfolder=subfolder, filename=relative_filename
                        )
                    download_list_item = {"filename": relative_path, "subfolder": subfolder, "size": file_size, "sha256": file_checksum, "url": url }
                    download_list.append( download_list_item )
                    print(download_list_item)

                print("total_file_list_size = ", total_file_list_size)


                # define a callback. returns True if download is cancelled.
                def bytes_downloaded_callback(total_bytes_downloaded, total_bytes_to_download):
                    total_bytes_to_download_gb = total_bytes_to_download / 1073741824.0
                    total_bytes_to_download_gb = f"{total_bytes_to_download_gb:.2f}"
                    total_bytes_downloadeded_gb = total_bytes_downloaded / 1073741824.0
                    total_bytes_downloadeded_gb = f"{total_bytes_downloadeded_gb:.2f}"
                    status = "Downloading... (" + total_bytes_downloadeded_gb + " / " + total_bytes_to_download_gb + ") GiB"
                    if total_bytes_to_download > 0:
                        self.model_install_status[model_id]["status"] = status
                        self.model_install_status[model_id]["percent"] = (total_bytes_downloaded / total_bytes_to_download) * 100.0

                        if "cancelled" in self.model_install_status[model_id]:
                            return True

                        return False

                total_bytes_downloaded = 0
                #okay, let's download the files one by one.
                for download_list_item in download_list:
                   local_filename = os.path.join(download_folder, download_list_item["filename"])

                   # create the subfolder (it may already exist, which is ok)
                   subfolder=os.path.join(download_folder, download_list_item["subfolder"])
                   os.makedirs(subfolder,  exist_ok=True)

                   print("Downloading", download_list_item["url"], " to ", local_filename)


                   downloaded_size = download_file_with_progress(download_list_item["url"], local_filename, bytes_downloaded_callback, total_bytes_downloaded, total_file_list_size)

                   if "cancelled" in self.model_install_status[model_id]:
                       return False

                   total_bytes_downloaded += downloaded_size

                return True
            except Exception as e:
                    print("Error retry:" + str(e))
                    retries_left -= 1
                    if "cancelled" in self.model_install_status[model_id]:
                       return False

        #we only get here if we failed (and exceeded max number of retries)
        return False

    # this combines the previous 'download_model' and 'download_quantized_models` routines into a single function (that rules them all)
    def _download_model(self, model_id):
        if model_id not in self.installable_model_map:
            print("Unexpected error! model_id=", model_id, " not found in installable_map!")
            return False

        installable_details = self.installable_model_map[model_id]

        if "repo_id" not in installable_details:
            print("Unexpected error! 'repo_id' key not found in installable_details for model_id=", model_id)
            return False

        if "supported_model_ids" not in installable_details:
            print("Unexpected error! 'supported_model_ids' key not found in installable_details for model_id=", model_id)
            return False

        if "download_exclude_filters" not in installable_details:
            print("Unexpected error! 'download_exclude_filters' key not found in installable_details for model_id=", model_id)
            return False

        repo_id = installable_details["repo_id"]

        if repo_id is None:
            print("Unexpected error! 'repo_id' value is None for model_id=", model_id)
            return False

        download_folder = 'hf_download_folder'

        download_exclude_filters = installable_details["download_exclude_filters"]

        download_success = self._download_hf_repo(repo_id, model_id, download_folder, download_exclude_filters)

        if download_success:

            # A given install_id may install multiple models. So, iterate through these.
            for supported_model in installable_details["supported_model_ids"]:

                # The 'supported_model' here, is the key to use for g_supported_model_map, which
                #  we will retrieve to get further installation location details.
                if supported_model not in g_supported_model_map:
                    print("Unexpected error! supported_model=", supported_model, " not found in supported model map. Installation model_id=", model_id)
                    return False

                supported_model_details = g_supported_model_map[supported_model]

                if "install_subdir" not in supported_model_details:
                    print("Unexpected error! 'install_subdir' not in supported_model_details for supported_model=", supported_model, ". Installation model_id=", model_id)
                    return False

                install_subdir = supported_model_details["install_subdir"]

                # We expect the subdir to have *at least* 2 entries.. e.g. ["stable-diffusion-ov", "some-model-specific-folder"],
                #  so double check that
                if len(install_subdir) < 2:
                    print("Unexpected error! 'install_subdir' for supported_model=", supported_model, " is array of less than 2 entries...")
                    return False

                full_install_path = os.path.join(self._weight_path, *install_subdir)

                # get 'right-most' folder in the subdir.
                leaf_folder = install_subdir[-1]

                # If <download_folder>/<leaf_folder> exists, then *that* is the folder we will copy
                #  to <full_install_path>
                # Otherwise, we will copy <download_folder> itself to <full_install_path>
                if os.path.isdir(os.path.join(download_folder, leaf_folder)):
                    copy_from_folder = os.path.join(download_folder, leaf_folder)
                else:
                    copy_from_folder = download_folder

                # double check that 'copy_from_folder' exists..
                if not os.path.isdir(copy_from_folder):
                    print("Unexpected error! copy_from_folder=", copy_from_folder, " doesn't exist.")
                    return False

                # double check that full_install_path is a subdirectory of our weight path.
                # (As we don't want to touch anything outside of our top-level 'weights' folder)
                if not is_subdirectory(full_install_path, self._weight_path):
                    print("Unexpected error! full_install_path=", full_install_path, " is not a subdirectory of the top-level weights folder.")
                    return False

                # if it already exists, delete it first
                if os.path.isdir(full_install_path):
                    shutil.rmtree(SD_path_INT8)

                # okay, copy it!
                # (the ignore_patterns were added to pull logic of 'download_quantized_models' into here. See below)
                shutil.copytree(copy_from_folder, full_install_path, ignore=shutil.ignore_patterns('FP16', 'INT8'))

                # The following logic was addeded so that we could call this function in place of 'download_quantized_models'
                # (as so much of the logic was the same.. just this INT8 / FP16 folder thing differed).
                # We should probably find a better way to handle it, but it beats having a completely separate function (I think)
                if os.path.isdir(os.path.join(download_folder, 'FP16')):
                    #if the 'right-most' installation directory *doesn't* contain 'int8',
                    # then this is the FP16 model.
                    if not 'int8' in leaf_folder:
                        # copy the FP16 collateral
                        shutil.copytree(os.path.join(download_folder, 'FP16'), full_install_path, dirs_exist_ok=True)
                    else:
                        if os.path.isdir(os.path.join(download_folder, 'INT8')):
                            # copy the INT8 collateral
                            shutil.copytree(os.path.join(download_folder, 'INT8'), full_install_path, dirs_exist_ok=True)

        # we redefine this, just to be sure it didn't somehow get changed/corrupted during the
        # download method.
        download_folder = 'hf_download_folder'
        if os.path.isdir(download_folder):
                shutil.rmtree(download_folder, ignore_errors=True)

        return True

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

                if "cancelled" in self.model_install_status[model_id]:
                    return

                #percent_complete += 0.2
                percent_complete += 1

                self.model_install_status[model_id]["percent"] = percent_complete

                if( percent_complete - last_perc_complete_printed > 10 ):
                    print("install_test %: ", percent_complete)
                    last_perc_complete_printed = percent_complete


    def cancel_install(self, model_id):
        print("cancel_install: model_id=", model_id)

        with self.model_install_status_lock:
            if model_id in self.model_install_status:
                self.model_install_status[model_id]["cancelled"] = True



    def install_model(self, model_id):
        print("install_model: model_id=", model_id)

        with self.model_install_status_lock:
            # set the status to 'Queued'. This is what will display in the UI
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

            # a list of installation id's where we only need to call
            # our download function (no fancy post-download processing required)
            simply_download_models = [
            "sd_15_portrait",
            "sd_15_landscape",
            "sd_15_inpainting",
            "sd_15_openpose",
            "sd_15_canny",
            "sd_15_scribble",
            "sd_15_Referenceonly"]

            if model_id in simply_download_models:
                self._download_model(model_id)
            elif  model_id == "sd_15_square":
                self.dl_sd_15_square(model_id)
            elif model_id == "sd_15_LCM":
                self.dl_sd_15_LCM(model_id)
            elif (model_id == "test1"):
                self.install_test(model_id)
            elif (model_id == "test2"):
                self.install_test(model_id)
            else:
                print("Warning! unknown model_id=", model_id)


            # Notify the next thread in the queue
            self.install_lock.notify_all()

        with self.model_install_status_lock:
            self.model_install_status.pop(model_id)



        print("install_model: model_id=", model_id, " done!")

    def dl_sd_15_square(self, model_id):
        print("Downloading Intel/sd-1.5-square-quantized Models")
        repo_id = "Intel/sd-1.5-square-quantized"
        model_fp16 = os.path.join("stable-diffusion-1.5", "square")
        model_int8 = os.path.join("stable-diffusion-1.5", "square_int8")
        install_location=self._install_location
        core = self._core
        npu_arch = self._npu_arch

        download_success = self._download_model(model_id)

        if npu_arch is not None:
            if download_success:
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
        compile_models = self._download_model(model_id)
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

                # get the list of installed models, and installable model details.
                installed_models, installable_model_details = ov_model_installer.get_all_model_details()

                # Send the list of installed models
                num_installed_models = len(installed_models)
                print("num_installed_models = ", num_installed_models)
                conn.sendall(bytes(str(num_installed_models), 'utf-8'))
                data = conn.recv(1024) # <-wait for ack
                for i in range(0, num_installed_models):
                    for detail in ["name", "id"]:
                        conn.sendall(bytes(installed_models[i][detail], 'utf-8'))
                        data = conn.recv(1024) # <-wait for ack

                # Send the installable model details
                num_installable_models = len(installable_model_details)
                print("num_installable_models = ", num_installable_models)
                conn.sendall(bytes(str(num_installable_models), 'utf-8'))
                data = conn.recv(1024) # <-wait for ack
                for i in range(0, num_installable_models):
                    for detail in ["name", "description", "id", "install_status"]:
                        conn.sendall(bytes(installable_model_details[i][detail], 'utf-8'))
                        data = conn.recv(1024) # <-wait for ack

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

            if data.decode() == "install_cancel":
                # send ack
                conn.sendall(data)

                # Get the model-id that we are interested in.
                data = conn.recv(1024)
                model_id = data.decode()

                #send ack
                conn.sendall(data)

                ov_model_installer.cancel_install(model_id)

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
