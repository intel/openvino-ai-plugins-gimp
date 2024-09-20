import os
import json
import sys
import traceback
import openvino as ov
from huggingface_hub import snapshot_download, HfApi, HfFileSystem, hf_hub_url
import concurrent.futures
import platform
import shutil
import io
import requests
import queue
import fnmatch
import logging
import threading
from pathlib import Path
from tqdm import tqdm
logging.basicConfig(format='%(message)s', level=logging.INFO, stream=sys.stdout)

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
        "name": "Stable Diffusion 1.5 [Square] [FP16]",
        "install_id": "sd_15_square",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "square"]
    },

    "sd_1.5_square_int8":
    {
        "name": "Stable Diffusion 1.5 [Square] [INT8]",
        "install_id": "sd_15_square",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"]
    },

    "sd_1.5_square_int8a16":
    {
        "name": "Stable Diffusion 1.5 [Square] [INT8A16]",
        "install_id": "sd_15_square",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"]
    },

    "sd_1.5_square_lcm":
    {
        "name": "Stable Diffusion 1.5 LCM [Square] [FP16]",
        "install_id": "sd_15_LCM",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_lcm"],
    },

    "sd_3.0_square":
    {
        "name": "Stable Diffusion 3.0 [Square]",
        "install_id": None, # Set to None, so that model manager UI doesn't give option to install.
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-3.0"],
    },

    "sd_1.5_portrait":
    {
        "name": "Stable Diffusion 1.5 [Portrait 360x640] [INT8]",
        "install_id": None,
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait"]
    },

    "sd_1.5_portrait_512x768":
    {
        "name": "Stable Diffusion 1.5 [Portrait 512x768] [INT8]",
        "install_id": None,
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait_512x768"],
    },

    "sd_1.5_landscape":
    {
        "name": "Stable Diffusion 1.5 [Landscape 640x360] [INT8]",
        "install_id": None,
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape"],
    },

    "sd_1.5_landscape_768x512":
    {
        "name": "Stable Diffusion 1.5 [Landscape 768x512] [INT8]",
        "install_id": None,
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape_768x512"],
    },

    "sd_1.5_inpainting":
    {
        "name": "Stable Diffusion 1.5 [Inpainting] [FP16]",
        "install_id": "sd_15_inpainting",
        "install_subdir": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting"],
    },

    "controlnet_openpose":
    {
        "name": "Stable Diffusion 1.5 [Controlnet OpenPose] [FP16]",
        "install_id": "sd_15_openpose",
        "install_subdir": ["stable-diffusion-ov", "controlnet-openpose"],
    },

    "controlnet_canny":
    {
        "name": "Stable Diffusion 1.5 [Controlnet Canny] [FP16]",
        "install_id": "sd_15_canny",
        "install_subdir": ["stable-diffusion-ov", "controlnet-canny"],
    },

    "controlnet_scribble":
    {
        "name": "Stable Diffusion 1.5 [Controlnet Scribble] [FP16]",
        "install_id": "sd_15_scribble",
        "install_subdir": ["stable-diffusion-ov", "controlnet-scribble"],
    },

    "controlnet_referenceonly":
    {
        "name": "Stable Diffusion 1.5 [Controlnet Reference-Only] [FP16]",
        "install_id": "sd_15_Referenceonly",
        "install_subdir": ["stable-diffusion-ov", "controlnet-referenceonly"],
    },

}


# add these to above dictionary for UI testing
'''
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
'''


# The thing used to populate the Model Manager UI (or model_setup.py console selections)
# Each model in g_supported_model_map defines 'install_id', which is the key to use.
#
# It's called 'base' model map, since it's meant to be
#  an initializer.. not something to use as-is. For example,
#  ModelManager makes a copy of it, and actually adds
#  more details to each entry.
# name: The name/title to be displayed in the Model Manager UI
# repo_id: HF repo id -- used by download routine.
# download_exclude_filters: an array of glob patterns. This is used to exclude downloading
#                           unnecessary files.
g_installable_base_model_map = {
    "sd_15_square":
    {
        "name": "Stable Diffusion 1.5 Square",
        "repo_id": "Intel/sd-1.5-square-quantized",
        "download_exclude_filters": ["*.blob"],
        "npu_compilation_routine": True,
    },

    "sd_15_LCM":
    {
        "name": "Stable Diffusion 1.5 LCM",
        "repo_id": "Intel/sd-1.5-lcm-openvino",
        "download_exclude_filters": ["*.blob", "unet_dynamic*"],
        "npu_compilation_routine": True,
    },

    "sd_15_portrait":
    {
        "name": "Stable Diffusion 1.5 Portrait",
        "repo_id": "Intel/sd-1.5-portrait-quantized",
        "download_exclude_filters": ["*.blob"],
    },

    "sd_15_landscape":
    {
        "name": "Stable Diffusion 1.5 Landscape",
        "repo_id": "Intel/sd-1.5-landscape-quantized",
        "download_exclude_filters": ["*.blob"],
    },

    "sd_15_inpainting":
    {
        "name": "Stable Diffusion 1.5 Inpainting",
        "repo_id": "Intel/sd-1.5-inpainting-quantized",
        "download_exclude_filters": ["*.blob", "INT8*"],
    },

    "sd_15_openpose":
    {
        "name": "Stable Diffusion 1.5 Controlnet: OpenPose",
        "repo_id": "Intel/sd-1.5-controlnet-openpose-quantized",
        "download_exclude_filters": ["*.blob", "INT8*"],
    },

    "sd_15_canny":
    {
        "name": "Stable Diffusion 1.5 Controlnet: Canny",
        "repo_id": "Intel/sd-1.5-controlnet-canny-quantized",
        "download_exclude_filters": ["*.blob", "INT8*"],
    },

    "sd_15_scribble":
    {
        "name": "Stable Diffusion 1.5 Controlnet: Scribble",
        "repo_id": "Intel/sd-1.5-controlnet-scribble-quantized",
        "download_exclude_filters": ["*.blob", "INT8*"],
    },

    "sd_15_Referenceonly":
    {
        "name": "Stable Diffusion 1.5 Controlnet: Reference-Only",
        "repo_id": "Intel/sd-reference-only",
        "download_exclude_filters": ["*.blob", "INT8*"],
    },

}

# add these to above dictionary for UI testing
'''
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

'''


access_token = None
# Constants
NPU_ARCH_3720 = "3720"
NPU_ARCH_4000 = "4000"
NPU_THRESHOLD = 43000

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
    try:
        # Compile the model for the specified device
        model = core.compile_model(model_path, device, config=config)

        # Export the compiled model to a binary blob
        with io.BytesIO() as model_blob:
            model.export_model(model_blob)

            # Write the binary blob to the output path
            temp_output_path = str(output_path) + ".tmp"
            with open(temp_output_path, 'wb') as f:
                f.write(model_blob.getvalue())

            # Remove the existing file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)

            # Rename the temporary file to the final output path
            os.rename(temp_output_path, output_path)

        logging.info(f"Model compiled and exported successfully to {output_path}")

    except Exception as e:
        logging.error(f"Failed to compile and export model: {str(e)}")
        tb_str = traceback.format_exc()
        raise RuntimeError(f"Model compilation or export failed for {model_path} on device {device}.\nDetails: {tb_str}")


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

def get_npu_architecture(core):
    try:
        available_devices = core.get_available_devices()
        if 'NPU' in available_devices:
            architecture = core.get_property('NPU', 'DEVICE_ARCHITECTURE')
            return NPU_ARCH_3720 if NPU_ARCH_3720 in architecture else NPU_ARCH_4000
    except Exception as e:
        logging.error(f"Error retrieving NPU architecture: {str(e)}")
        return None


def get_npu_driver_version(core):
    try:
        available_devices = core.get_available_devices()
        if 'NPU' in available_devices:
            driver_version = str(core.get_property('NPU', 'NPU_DRIVER_VERSION'))
            return driver_version
    except Exception as e:
        logging.error(f"Error retrieving NPU driver version: {str(e)}")
        return None

    return None


def get_npu_config(core, architecture):
    try:
        if architecture == NPU_ARCH_4000:
            gops_value = core.get_property("NPU", "DEVICE_GOPS")[ov.Type.i8]
            return 6 if gops_value > NPU_THRESHOLD else None
    except Exception as e:
        logging.error(f"Error retrieving NPU configuration: {str(e)}")
        return None

class ModelManager:
    def __init__(self, weight_path):
        self._core = ov.Core()
        self._npu_arch = get_npu_architecture(self._core)
        self._npu_config = get_npu_config(self._core, self._npu_arch)
        self._npu_driver_version = get_npu_driver_version(self._core)
        self._weight_path = weight_path
        self._install_location = os.path.join(self._weight_path, "stable-diffusion-ov")

        self.show_hf_download_tqdm = False

        self.hf_api = HfApi()
        self.hf_fs = HfFileSystem()

        # This is a map from model_install_id -> install progress details.
        # If a particular model_install_id is not present in the map, then it's not currently getting installed.
        self.model_install_status = {}
        self.model_install_status_lock = threading.Lock()

        # If an error occurs during install, the installation routine will write details of the error to this map.
        # If a particular model_install_id is not present in the map, then no error occurred during install.
        # The values of this map is a dictionary with the following entries:
        # "summary": "A high level description of the error. e.g. 'Download Failed', 'NPU Compilation Routine Failed', etc."
        # "details": "Some detailed description, usually a traceback. Something meaningful to a developer"
        self.model_install_error_condition = {}

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

    def get_error_details(self, model_id):
        summary = "None"
        details = "None"

        if model_id in self.model_install_error_condition:
            if "summary" in self.model_install_error_condition[model_id]:
                summary = self.model_install_error_condition[model_id]["summary"]
            if "details" in self.model_install_error_condition[model_id]:
                details = self.model_install_error_condition[model_id]["details"]

            # we only allow the client to get the error details once, and then we clear it automatically
            self.model_install_error_condition.pop(model_id)

        return summary, details

    # Given a model_id, read install_info.json from the installed directory (if it exists).
    # Returns the dictionary of info if the file exists.
    # Otherwise, returns None
    def get_installed_info(self, model_id):
        try:
            if model_id not in g_supported_model_map:
                return None

            install_subdir = g_supported_model_map[model_id]["install_subdir"]

            full_install_path = os.path.join(self._weight_path, *install_subdir)

            if not os.path.isdir(full_install_path):
                return None

            json_path = os.path.join(full_install_path, "install_info.json")

            if not os.path.isfile(json_path):
                return None

            with open(json_path, "r") as file:
                installed_info = json.load(file)

            return installed_info

        except Exception as e:
            print(f"Exception in get_installed_info(.., {model_id})")
            traceback.print_exc()
            return None

    # Given a model_id, is it installed? This is a fairly simple check right now.
    # Basically, if the install directory exists, and install_info.json exists within
    # it, then we consider the model to be installed.
    def is_model_installed(self, model_id):
        try:
            installed_info = self.get_installed_info(model_id)

            if installed_info is None:
                return False

            # sd_1.5_square_int8a16 is a newer model that has same install directory as sd_1.5_square / sd_1.5_square_int8
            #  For this model, we add a specific check for one of the files to make sure this model has actually been installed.
            if model_id == "sd_1.5_square_int8a16":
                install_subdir = g_supported_model_map[model_id]["install_subdir"]
                full_install_path = os.path.join(self._weight_path, *install_subdir)
                required_bin_path = os.path.join(full_install_path, "unet_int8a16.bin")
                if not os.path.isfile(required_bin_path):
                    print(f"{model_id} installation folder exists, but it is missing {required_bin_path}")
                    return False

            return True

        except Exception as e:
            print(f"Exception in is_model_installed(.., {model_id}")
            traceback.print_exc()
            return False

    # Given a model_id, is there an update available? This is what is used to determine whether installted models
    #  show up as a greyed-out 'Installed' button, or as a clickable 'Update' button.
    # Right now we only return True if we detect that a driver update has occurred since the model was installed.
    def is_model_update_available(self, model_id):
        try:

            # For platforms without NPUs, no updates are necessary (for now).
            if self._npu_driver_version is None:
                return False

            installed_info = self.get_installed_info(model_id)

            if installed_info is None:
                return False


            install_id = g_supported_model_map[model_id]["install_id"]

            if install_id is None:
                return False

            installable_map_entry = self.installable_model_map[install_id]

            # If this install routine has no NPU compilation routine, return False
            if "npu_compilation_routine" not in installable_map_entry:
                return False

            if installable_map_entry["npu_compilation_routine"] is not True:
                return False

            # Kind of a weird case -- the model was installed, but somehow the NPU compilation routine was skipped.
            # In this case, return True to give user the ability to click 'Update'
            if "npu_blob_driver_version" not in installed_info:
                return True

            installed_npu_driver_version = str(installed_info["npu_blob_driver_version"])
            current_system_npu_driver_version = str(self._npu_driver_version)

            # If these NPU driver versions do not match, then it means that the driver has been updated since these blobs were installed.
            if installed_npu_driver_version != current_system_npu_driver_version:
                print(f"is_model_update_available: model_id={model_id}, returning True because NPU driver update detected {installed_npu_driver_version} -> {current_system_npu_driver_version}")
                return True

            return False

        except Exception as e:
            print(f"Exception in is_model_update_available(.., {model_id}")
            traceback.print_exc()
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

                # If this install_id doesn't have any supported model id's,
                #  don't populate the model UI with this option.
                if len(install_details["supported_model_ids"]) == 0:
                    continue

                all_are_installed = True

                if install_id in self.model_install_error_condition:
                    install_status = "install_error"
                else:
                    for model_id in install_details["supported_model_ids"]:
                        if self.is_model_installed(model_id) is False:
                            all_are_installed = False
                            break

                    if all_are_installed:
                        install_status = "installed"

                        any_updates_available = False

                        #Now, check if there are any updates available..
                        for model_id in install_details["supported_model_ids"]:
                            if self.is_model_update_available(model_id) is True:
                                any_updates_available = True
                                break

                        if any_updates_available is True:
                            install_status = "installed_updates_available"

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


    # Download a given HF repo to the given download_folder.
    # This function returns true if the download was cancelled, otherwise it returns False upon success.
    # All errors are raised as exceptions, so it's recommended to wrap this in a try/except clause.
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

                # get the commit-id for the huggingface repo that we will be downloading files from.
                repo_info = self.hf_api.repo_info(repo_id)
                commit_id = repo_info.sha

                # save the HF repo information to the install_info for this model_id. This information will
                #  get written into a json file at the end of the install procedure.
                self.model_install_status[model_id]["install_info"]["hf_repo_id"] = repo_id
                self.model_install_status[model_id]["install_info"]["hf_commit_id"] = commit_id

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
                            #print(relative_path, ": Skipped due to exclude filters")
                            continue

                    total_file_list_size += file_size

                    subfolder = os.path.dirname(relative_path).replace("\\", "/")
                    relative_filename = os.path.basename(relative_path)
                    url = hf_hub_url(
                            repo_id=repo_id, subfolder=subfolder, filename=relative_filename
                        )
                    download_list_item = {"filename": relative_path, "subfolder": subfolder, "size": file_size, "sha256": file_checksum, "url": url }
                    download_list.append( download_list_item )
                    #print(download_list_item)

                #print("total_file_list_size = ", total_file_list_size)


                if self.show_hf_download_tqdm is True:
                    bar_format = '{desc}: |{bar}| {percentage:3.0f}% [elapsed: {elapsed}, remaining: {remaining}]'
                    total_bytes_to_download_gb = total_file_list_size / 1073741824.0
                    total_bytes_to_download_gb = f"{total_bytes_to_download_gb:.2f}"
                    progress_bar = tqdm(total = 1000, desc=f'Downloading Repo ({repo_id}, {total_bytes_to_download_gb} GiB)', bar_format=bar_format)


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

                        if self.show_hf_download_tqdm is True:
                           progress_bar.n = self.model_install_status[model_id]["percent"] * 10
                           progress_bar.refresh()

                        if "cancelled" in self.model_install_status[model_id]:
                            return True

                        # False means 'not cancelled'
                        return False

                total_bytes_downloaded = 0
                #okay, let's download the files one by one.
                for download_list_item in download_list:
                   local_filename = os.path.join(download_folder, download_list_item["filename"])

                   # create the subfolder (it may already exist, which is ok)
                   subfolder=os.path.join(download_folder, download_list_item["subfolder"])
                   os.makedirs(subfolder,  exist_ok=True)

                   #print("Downloading", download_list_item["url"], " to ", local_filename)
                   downloaded_size = download_file_with_progress(download_list_item["url"], local_filename, bytes_downloaded_callback, total_bytes_downloaded, total_file_list_size)

                   if "cancelled" in self.model_install_status[model_id]:
                       return True

                   total_bytes_downloaded += downloaded_size

                return False
            except Exception as e:
                    print("Error retry:" + str(e))
                    retries_left -= 1
                    if "cancelled" in self.model_install_status[model_id]:
                       return True

                    if retries_left <= 0:
                        raise e

        # There shouldn't be any way to get here..
        raise RuntimeError(f"Unexpected exit from _download_hf_repo..")

    # this combines the previous 'download_model' and 'download_quantized_models` routines into a single function (that rules them all)
    # Returns True if the download was successful. Otherwise (error or cancellation), it returns False.
    def _download_model(self, model_id):
        try:
            if model_id not in self.installable_model_map:
                raise RuntimeError(f"Error! model_id={model_id} not found in installable_map!")

            installable_details = self.installable_model_map[model_id]

            if "repo_id" not in installable_details:
                raise RuntimeError(f"Error! 'repo_id' key not found in installable_details for model_id={model_id}")

            if "supported_model_ids" not in installable_details:
                raise RuntimeError(f"Error! 'supported_model_ids' key not found in installable_details for model_id={model_id}")

            if "download_exclude_filters" not in installable_details:
                raise RuntimeError(f"Error! 'download_exclude_filters' key not found in installable_details for model_id={model_id}")

            repo_id = installable_details["repo_id"]

            if repo_id is None:
                raise RuntimeError(f"Error! 'repo_id' value is None for model_id={model_id}")

            download_folder = 'hf_download_folder'

            download_exclude_filters = installable_details["download_exclude_filters"]

            # download the hf repo.
            # Note that by design, we want exceptions to be caught by outer try/catch in this function...
            #  so don't be tempted to add another here..
            download_cancelled = self._download_hf_repo(repo_id, model_id, download_folder, download_exclude_filters)

            if not download_cancelled:

                # A given install_id may install multiple models. So, iterate through these.
                for supported_model in installable_details["supported_model_ids"]:

                    # The 'supported_model' here, is the key to use for g_supported_model_map, which
                    #  we will retrieve to get further installation location details.
                    if supported_model not in g_supported_model_map:
                        raise RuntimeError(f"Error! supported_model={supported_model} not found in supported model map. Installation model_id={model_id}")

                    supported_model_details = g_supported_model_map[supported_model]

                    if "install_subdir" not in supported_model_details:
                        raise RuntimeError(f"Error! 'install_subdir' not in supported_model_details for supported_model={supported_model}. model_id={model_id}")

                    install_subdir = supported_model_details["install_subdir"]

                    # We expect the subdir to have *at least* 2 entries.. e.g. ["stable-diffusion-ov", "some-model-specific-folder"],
                    #  so double check that
                    if len(install_subdir) < 2:
                        raise RuntimeError(f"Error! 'install_subdir' for supported_model={supported_model}. is array of less than 2 entries...")

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
                        raise RuntimeError(f"Error! copy_from_folder={copy_from_folder} doesn't exist.")

                    # double check that full_install_path is a subdirectory of our weight path.
                    # (As we don't want to touch anything outside of our top-level 'weights' folder)
                    if not is_subdirectory(full_install_path, self._weight_path):
                        raise RuntimeError(f"Error! full_install_path={full_install_path} is not a subdirectory of the top-level weights folder.")

                    # if it already exists, delete it first
                    if os.path.isdir(full_install_path):
                        shutil.rmtree(full_install_path)

                    # okay, copy it!
                    # (the ignore_patterns were added to pull logic of 'download_quantized_models' into here. See below)
                    shutil.copytree(copy_from_folder, full_install_path, ignore=shutil.ignore_patterns('FP16', 'INT8'))

                    # The following logic was addeded so that we could call this function in place of 'download_quantized_models'
                    # (as so much of the logic was the same.. just this INT8 / FP16 folder thing differed).
                    # We should probably find a better way to handle it, but it beats having a completely separate function (I think)
                    # hmm, double check INT8A16... I think it may just get copied twice in this case?
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

            if download_cancelled:
                return False

        except Exception as e:
            # print it:
            traceback.print_exc()

            # .. but also capture it as a string
            tb_str = traceback.format_exc()

            self.model_install_error_condition[model_id] = {}
            self.model_install_error_condition[model_id]["summary"] = "Model Download Routine Failed"
            self.model_install_error_condition[model_id]["details"] = tb_str

            return False

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
        #print("install_model: model_id=", model_id)
        try:
            with self.model_install_status_lock:
                # Populate the initial install status dictionary, and set the status to 'Queued'.
                # This is what will display in the UI until it's this model's turn to get installed.
                self.model_install_status[model_id] = {"status": "Queued", "percent": 0.0, "install_info": {}}

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

                only_npu_recompilation = False

                #iterate through all models for this install_id, and determine whether
                # they are all installed.
                all_are_installed = True
                install_details = self.installable_model_map[model_id]
                for ind_model_id in install_details["supported_model_ids"]:
                        if self.is_model_installed(ind_model_id) is False:
                            all_are_installed = False
                            break

                # If all models are already installed, then this must be a request to 'update',
                #  which means NPU Recompilation.
                if all_are_installed:
                    if self._npu_arch is not None:
                        print(f"{model_id}: Recompiling NPU models") 
                    else:
                        print(f"{model_id}: Model is already installed. Skipping download. ") 

                    # we still need to set this to avoid re-downloading everything. 
                    only_npu_recompilation = True

                    #Initialize the install_info with the existing one,
                    # since in this mode we won't be downloading anything.. we
                    # want to preserve HF commit-id, etc.
                    installed_info = self.get_installed_info(install_details["supported_model_ids"][0])
                    self.model_install_status[model_id]["install_info"] = installed_info


                if model_id in simply_download_models:
                    if only_npu_recompilation is False:
                        self._download_model(model_id)
                    else:
                        print("only_npu_recompilation is unexpectedly set to True for a 'download-only' model.. skipping")
                elif  model_id == "sd_15_square":
                    self.dl_sd_15_square(model_id, only_npu_recompilation)
                elif model_id == "sd_15_LCM":
                    self.dl_sd_15_LCM(model_id, only_npu_recompilation)
                elif (model_id == "test1"):
                    self.install_test(model_id)
                elif (model_id == "test2"):
                    self.install_test(model_id)
                else:
                    print("Warning! unknown model_id=", model_id)


                # if the installation was not cancelled..
                if "cancelled" not in self.model_install_status[model_id]:
                    try:
                        # If there was not an error in installation, write some installation info to the install directory.
                        if model_id not in self.model_install_error_condition:
                            for supported_model in self.installable_model_map[model_id]["supported_model_ids"]:
                                install_subdir = g_supported_model_map[supported_model]["install_subdir"]
                                full_install_path = os.path.join(self._weight_path, *install_subdir)

                                if is_subdirectory(full_install_path, self._weight_path):
                                    # If the install info dictionary is non-empty, write the info.
                                    if self.model_install_status[model_id]["install_info"]:
                                        file_name = "install_info.json"
                                        with open(os.path.join(full_install_path, file_name), 'w') as json_file:
                                            json.dump(self.model_install_status[model_id]["install_info"], json_file, indent=4)
                    except Exception as e:
                        # print it:
                        traceback.print_exc()

                        # .. but also capture it as a string
                        tb_str = traceback.format_exc()

                        self.model_install_error_condition[model_id] = {}
                        self.model_install_error_condition[model_id]["summary"] = "Post Install Routine Failed"
                        self.model_install_error_condition[model_id]["details"] = tb_str

                # Notify the next thread in the queue
                self.install_lock.notify_all()

            with self.model_install_status_lock:
                self.model_install_status.pop(model_id)

        except Exception as e:
            # note, this should only happen if there is an exception thrown *inside* of this
            # this function.
            print("Exception within install_model routine:")
            # print it:
            traceback.print_exc()

            # .. but also capture it as a string
            tb_str = traceback.format_exc()

            if model_id not in self.model_install_error_condition:
                self.model_install_error_condition[model_id] = {}
                self.model_install_error_condition[model_id]["summary"] = "General Install Routine Failed"
                self.model_install_error_condition[model_id]["details"] = tb_str

        print("Done!")

    def dl_sd_15_square(self, model_id, only_npu_recompilation=False):
        repo_id = "Intel/sd-1.5-square-quantized"
        model_fp16 = os.path.join("stable-diffusion-1.5", "square")
        model_int8 = os.path.join("stable-diffusion-1.5", "square_int8")
        install_location=self._install_location
        core = self._core
        npu_arch = self._npu_arch
        npu_config = self._npu_config

        download_success = True

        # If we are only recompiling the NPU models, don't download.
        if only_npu_recompilation is False:
            print("Downloading Intel/sd-1.5-square-quantized Models")
            download_success = self._download_model(model_id)
        
        if npu_arch is not None:
            if download_success:
                try:
                    self.model_install_status[model_id]["status"] = "Compiling models for NPU..."
                    self.model_install_status[model_id]["percent"] = 0.0
                    text_future = None
                    unet_int8_future = None
                    unet_int8a16_future = None
                    unet_future = None
                    vae_de_future = None
                    vae_en_future = None

                    if npu_arch == NPU_ARCH_3720:
                        # larger model should go first to avoid multiple checking when the smaller models loaded / compiled first
                        models_to_compile = [ "unet_int8a16", "unet_int8", "unet_bs1", "text_encoder"]
                        shared_models = ["text_encoder.blob"]
                        sd15_futures = {
                            "text_encoder" : text_future,
                            "unet_int8" : unet_int8_future,
                            "unet_int8a16" : unet_int8a16_future,
                            "unet_bs1" : unet_future,
                            
                        }
                    else:
                        # also modified the model order for less checking in the future object when it gets result
                        models_to_compile = [ "unet_int8a16", "unet_int8", "unet_bs1", "text_encoder", "vae_encoder" , "vae_decoder" ]
                        shared_models = ["text_encoder.blob", "vae_encoder.blob", "vae_decoder.blob"]
                        sd15_futures = {
                            "text_encoder" : text_future,
                            "unet_bs1" : unet_future,
                            "unet_int8a16" : unet_int8a16_future,
                            "unet_int8" : unet_int8_future,
                            "vae_encoder" : vae_en_future,
                            "vae_decoder" : vae_de_future
                        }


                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                        for model_name in models_to_compile:
                            config = None
                            logging.info(f"Creating NPU model for {model_name}")

                            if "unet_int8" in model_name or "unet_bs1" in model_name:
                                config = { "NPU_DPU_GROUPS" : npu_config, "NPU_MAX_TILES": npu_config } if npu_config is not None else None

                            if "unet_int8" not in model_name:
                                model_path_fp16 = os.path.join(install_location, model_fp16, model_name + ".xml")
                                output_path_fp16 = os.path.join(install_location, model_fp16, model_name + ".blob")
                                sd15_futures[model_name] = executor.submit(compile_and_export_model, core, model_path_fp16, output_path_fp16, config=config)
                            else:
                                model_path_int8 = os.path.join(install_location, model_int8, model_name + ".xml")
                                output_path_int8 = os.path.join(install_location, model_int8, model_name + ".blob")
                                sd15_futures[model_name] = executor.submit(compile_and_export_model, core, model_path_int8, output_path_int8, config=config)


                        num_futures = len(sd15_futures)
                        perc_increment = 100.0 / num_futures

                        self.model_install_status[model_id]["percent"] = 0.0
                        for model_name, model_future in sd15_futures.items():
                            model_future.result()
                            self.model_install_status[model_id]["percent"] += perc_increment


                    # Copy shared models to INT8 directory
                    for blob_name in shared_models:
                        shutil.copy(
                            os.path.join(install_location, model_fp16, blob_name),
                            os.path.join(install_location, model_int8, blob_name)
                        )

                    # Record the npu_driver version that we used to create the blobs.
                    self.model_install_status[model_id]["install_info"]["npu_blob_driver_version"] = self._npu_driver_version

                    config_fp_16 = { 	"power modes supported": "yes",
                                            "best performance" : ["GPU","GPU","GPU","GPU"],
                                                    "balanced" : ["NPU","NPU","GPU","GPU"],
                                       "best power efficiency" : ["NPU","NPU","NPU","GPU"]
                    }
                    config_int8 = { 	"power modes supported": "yes",
                                            "best performance" : ["NPU","NPU","GPU","GPU"],
                                                    "balanced" : ["GPU","NPU","NPU","GPU"],
                                       "best power efficiency" : ["NPU","NPU","NPU","GPU"]
                    }

                    # Specify the file name
                    file_name = "config.json"

                    # Write the data to a JSON file
                    with open(os.path.join(install_location, model_fp16, file_name), 'w') as json_file:
                        json.dump(config_fp_16, json_file, indent=4)
                    # Write the data to a JSON file
                    with open(os.path.join(install_location, model_int8, file_name), 'w') as json_file:
                        json.dump(config_int8, json_file, indent=4)

                except Exception as e:
                    # print it:
                    traceback.print_exc()

                    # .. but also capture it as a string
                    tb_str = traceback.format_exc()

                    self.model_install_error_condition[model_id] = {}
                    self.model_install_error_condition[model_id]["summary"] = "NPU Compilation Routine Failed"
                    self.model_install_error_condition[model_id]["details"] = tb_str


    def dl_sd_15_LCM(self, model_id, only_npu_recompilation=False):
        repo_id = "Intel/sd-1.5-lcm-openvino"
        model_1 = "square_lcm"
        model_2 = None

        install_location=self._install_location
        core = self._core
        npu_arch = self._npu_arch

        download_success = True

        # If we are only recompiling the NPU models, don't download.
        if only_npu_recompilation is False:
            print("Downloading Intel/sd-1.5-lcm-openvino")
            download_success = self._download_model(model_id)

        if npu_arch is not None:
            if download_success:
                try:
                    self.model_install_status[model_id]["status"] = "Compiling models for NPU..."
                    self.model_install_status[model_id]["percent"] = 0.0
                    text_future = None
                    unet_future = None
                    vae_de_future = None

                    if npu_arch == NPU_ARCH_3720:
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

                    # Save the npu_driver version that we used to create the blobs.
                    self.model_install_status[model_id]["install_info"]["npu_blob_driver_version"] = self._npu_driver_version

                    # Write out config file. GPU used for VAE in all cases.
                    config = { 	"power modes supported": "yes",
                                    "best performance" : ["GPU","GPU","GPU"],
                                            "balanced" : ["GPU","NPU","GPU"],
                               "best power efficiency" : ["NPU","NPU","GPU"]
                        }

                    # Specify the file name
                    file_name = "config.json"

                    # Write the data to a JSON file
                    with open(os.path.join(install_location, "stable-diffusion-1.5", model_1, file_name), 'w') as json_file:
                        json.dump(config, json_file, indent=4)
                except Exception as e:
                    # print it:
                    traceback.print_exc()

                    # .. but also capture it as a string
                    tb_str = traceback.format_exc()

                    self.model_install_error_condition[model_id] = {}
                    self.model_install_error_condition[model_id]["summary"] = "NPU Compilation Routine Failed"
                    self.model_install_error_condition[model_id]["details"] = tb_str