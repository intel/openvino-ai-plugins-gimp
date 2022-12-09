import os
#import pickle


def get_weight_path():
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, "gimp_openvino_config.txt"), "r") as file:
        for line in file.readlines():
            if line.split("=")[0] == "weight_path":
                weight_path = line.split("=")[1].replace("\n", "")
                #print("weight_path in program",weight_path)
                return weight_path
