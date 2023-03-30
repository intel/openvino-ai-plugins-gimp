# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

"""
Script will download weights and create gimp_openvino_config.txt, and print path to be added to GIMP Preferences.
"""
import os
import sys
import json
#import pickle
import csv
import hashlib
import gdown
import gimpopenvino
from openvino.runtime import Core


def setup_python_weights(install_location=None):
    if not install_location:
        install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp")
        
    if not os.path.isdir(install_location):
        os.mkdir(install_location)
    python_string = "python"
    if os.name == "nt":  # windows
        python_string += ".exe"
    python_path = os.path.join(os.path.dirname(sys.executable), python_string)
    weight_path = os.path.join(install_location, "weights")
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)

    step = 1
    print("\n##########")
    if os.name == "nt":  # windows
        print(
            "{}>> Reference Models present in weights folder. You can also download them from open model zoo.".format(
                step
            )
        )
        step += 1
        print(
            "{}>> Please move the weights folder from the cloned repository: \n"
            "GIMP-OV".format(
                step
            )
        )
        print("and place in: " + weight_path)
        step += 1
    else:  # linux
        file_path = os.path.join(os.path.dirname(gimpopenvino.__file__), "tools")
        with open(os.path.join(file_path, "model_info.csv")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            headings = next(csv_reader)
            for row in csv_reader:
                model_path, file_id = os.path.join(*row[0].split("/")), row[1]
                file_size, model_file_name, md5sum = float(row[2]), row[3], row[4]
                if not os.path.isdir(os.path.join(weight_path, model_path)):
                    os.makedirs(os.path.join(weight_path, model_path))
                destination = os.path.join(
                    os.path.join(weight_path, model_path), model_file_name
                )
 
                if not os.path.isfile(destination) or (digest and digest != md5sum):
                    print(
                        "\nDownloading "
                        + model_path
                        + "(~"
                        + str(file_size)
                        + "MB)..."
                    )
                    url = "https://drive.google.com/uc?id={0}".format(file_id)
                    try:
                        gdown.cached_download(url, destination, md5=md5sum)
                    except:
                        try:
                            gdown.download(url, destination, quiet=False)
                        except:
                            print("Failed to download !")

    plugin_loc = os.path.dirname(gimpopenvino.__file__)
    ie = Core()
    supported_devices = ie.available_devices
    py_dict = {
        "python_path" : python_path,
        "weight_path" : weight_path,
        "supported_devices": supported_devices
        }
    with open(os.path.join(plugin_loc, "tools", "gimp_openvino_config.json"), "w") as file:
        json.dump(py_dict,file)


    print(
        "{}>> Please add this path to Preferences --> Plug-ins in GIMP : ".format(step),
        os.path.join(plugin_loc, "plugins"),
    )
    print("##########\n")


if __name__ == "__main__":
    setup_python_weights()
