#!/usr/bin/env python3
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Script to create and configure gimp_openvino_config.json
"""
import os
import re
import sys
import json
import uuid
import shutil
import platform
import subprocess
from pathlib import Path

import gimpopenvino
import openvino as ov 
from gimpopenvino.plugins.openvino_utils.tools.tools_utils import base_model_dir, config_path_dir


def install_base_models(base_model_dir, repo_weights_dir):
    for folder in os.scandir(repo_weights_dir):
        model_name = os.path.basename(folder)
        model_path = os.path.join(base_model_dir, model_name)
        if not os.path.isdir(model_path):
            print(f"Copying {model_name} to {base_model_dir}")
            shutil.copytree(Path(folder), model_path)
    print("Setup done for base models.")


def filter_supported_devices(core):
    ZERO_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")
    valid_devices = []

    for device in core.get_available_devices():
        full_name = core.get_property(device, "FULL_DEVICE_NAME")
        supported_props = core.get_property(device, "SUPPORTED_PROPERTIES")

        # Skip if not Intel
        if "Intel" not in full_name:
            continue

        # skip device if it is NPU < 3720
        if "AI Boost" in full_name and "AUTO_DETECT"  in core.get_property(device, "DEVICE_ARCHITECTURE"):
            if  core.get_property(device, "DEVICE_GOPS")[ov.Type.i8] == 0:
                continue

        # Skip if device has a zero UUID (indicates a non-unique or invalid device)
        if "DEVICE_UUID" in supported_props:
            dev_uuid = uuid.UUID(core.get_property(device, "DEVICE_UUID"))
            if dev_uuid == ZERO_UUID:
                continue

        valid_devices.append(device)

    return valid_devices

def get_plugin_version(file_dir=None):
    """
    Retrieves the plugin version via git tags if available, ensuring
    the command is run from the directory where this Python file resides.

    Returns:
        str: Plugin version from git or "0.0.0dev0" if git is unavailable.

    Why use git describe for this? Because generates a human-readable string to 
    identify a particular commit in a Git repository, using the closest (most recent) 
    annotated tag reachable from that commit. Typically, it looks like:
    <tag>[-<number_of_commits_since_tag>-g<abbreviated_commit_hash>]
    
    For example, if your commit is exactly tagged 1.0.0, running 
    git describe might simply return 1.0.0. If there have been 10 
    commits since the v1.0.0 tag, git describe might return something like:
    1.0.0-10-g3ab12ef
    where:

    1.0.0 is the closest tag in the commit history.
    10 is how many commits you are ahead of that tag.
    g3ab12ef is the abbreviated hash of the current commit.

    we can then turn this into a PEP440 compliant string
    """
    try:
        raw_version = subprocess.check_output(
            ["git", "describe", "--tags"],
            cwd=file_dir,
            encoding="utf-8"
        ).strip()
        
        # Normalize the git version to PEP 440
        match = re.match(r"v?(\d+\.\d+\.\d+)(?:-(\d+)-g[0-9a-f]+)?", raw_version)

        if match:
            version, dev_count = match.groups()
            if dev_count:
                return f"{version}.dev{dev_count}"  # PEP 440 dev version
            return version
        else:
            raise ValueError(f"Invalid version format: {raw_version}")
    except Exception as e:
        print(f"Error obtaining version: {e}")
        return "0.0.0"  # Fallback version    


def complete_install(install_location=None, repo_weights_dir=None):
    # Determine the AI model install location
    if not install_location:
        install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp")

    # Create the install directory if it doesn't exist
    os.makedirs(install_location, exist_ok=True)

    # Determine Python executable path
    python_path = sys.executable

    # Create the weights directory if it doesn't exist
    weight_path = os.path.join(install_location, "weights")
    os.makedirs(weight_path, exist_ok=True)

    plugin_version = "Unknown"
    # Install base models from a repo and get the plugin version number.
    if repo_weights_dir:
        install_base_models(weight_path, repo_weights_dir)
        plugin_version = get_plugin_version(repo_weights_dir)

    # print("\n##########\n")

    # Determine plugin location (where gimpopenvino is installed)
    plugin_loc = os.path.dirname(gimpopenvino.__file__)

    # Filter supported devices using OpenVINO runtime
    core = ov.Core()
    supported_devices = filter_supported_devices(core)

    # Build the JSON config data
    py_dict = {
        "python_path": python_path,
        "weight_path": weight_path,
        "supported_devices": supported_devices,
        "plugin_version": plugin_version
    }

    # Write config data to gimp_openvino_config.json
    govconfig_path = os.path.join(config_path_dir, "gimp_openvino_config.json")
    with open(govconfig_path, "w+") as file:
        json.dump(py_dict, file)

    # For Linux, add executable permissions to plugin scripts
    if platform.system() == "Linux":
        scripts = [
            "plugins/superresolution_ov/superresolution_ov.py",
            "plugins/stable_diffusion_ov/stable_diffusion_ov.py",
            "plugins/semseg_ov/semseg_ov.py",
        ]
        for script in scripts:
            script_path = os.path.join(plugin_loc, script)
            subprocess.call(["chmod", "+x", script_path])
