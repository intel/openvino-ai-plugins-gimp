import os
import json
#import pickle


def get_weight_path():
    config_path = os.path.dirname(os.path.realpath(__file__))
    #data={}
    with open(os.path.join(config_path, "gimp_openvino_config.json"), "r") as file:
        data = json.load(file)
        #print("data",data)
    #python_path=data["python_path"]
    weight_path=data["weight_path"]
    #print("python_path:",python_path)
    #print("weight_path:",weight_path)
    return weight_path

    #with open(os.path.join(config_path, "gimp_openvino_config.txt"), "r") as file:
    #    for line in file.readlines():
    #        if line.split("=")[0] == "weight_path":
    #            weight_path = line.split("=")[1].replace("\n", "")
                #print("weight_path in program",weight_path)
    #            return weight_path

if __name__ == "__main__":
    wgt = get_weight_path()
    #print("wgt", wgt)
