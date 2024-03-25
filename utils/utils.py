import os
import random

import numpy as np
import torch
import utils.preprocess as preprocess
import yaml
from termcolor import colored


def set_seed(seed=1024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize(array):
    # Calculate the minimum and maximum values along axis 0 (columns)
    min_vals = np.min(array, axis=0)
    max_vals = np.max(array, axis=0)

    # Normalize the array to the range [0, 1] along each column
    normalized_array = (array - min_vals) / (max_vals - min_vals)
    return normalized_array


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config


def build_config(config):
    config = load_config(config)
    config["DEVICE"] = torch.device(config["DEVICE"])

    # set number of classes
    config_data = config["dataset"]
    config_data["name"] = config_data["data_root"].split("/")[-1].lower()
    if "office31" in config_data["name"]:
        config["Architecture"]["class_num"] = 31
        
    elif "officehome" in config_data["name"]:
        config["Architecture"]["class_num"] = 65
        
    elif "domainnet" in config_data["name"] and config_data["method"] == "SSDA":
        config["Architecture"]["class_num"] = 126
        
    elif "domainnet" in config_data["name"] and config_data["method"] == "UDA":
        config["Architecture"]["class_num"] = 40
        
    elif "pacs" in config_data["name"]:
        config["Architecture"]["class_num"] = 7
        
    elif "geo" in config_data["name"]:
        config["Architecture"]["class_num"] = 600
        
    elif "visda" in config_data["name"]:
        config["Architecture"]["class_num"] = 12
        
    else:
        raise ValueError(
            "Dataset cannot be recognized. Please define your own dataset here."
        )

    ### DATA AUGMENTATION - ASDA ###
    config_data["prep"] = {
        "source_w": preprocess.train_weak_asda(**config_data["prep"]["params"]),
        "source_str": preprocess.source_train_strong_asda(
            **config_data["prep"]["params"]
        ),
        "target_w": preprocess.train_weak_asda(**config_data["prep"]["params"]),
        "target_str": preprocess.target_train_strong_asda(
            **config_data["prep"]["params"]
        ),
        "test": preprocess.image_test_asda(**config_data["prep"]["params"]),
        "val": preprocess.image_val_asda(**config_data["prep"]["params"]),
    }
    ################################

    # create output folder and log file
    source_name = config_data["source"]["name"]
    target_name = config_data["target"]["name"]
    target_shot = config_data["target_shot"]
    method = (
        config_data["method"]
        if config_data["method"] == "UDA"
        else f"{config_data['method']}_{target_shot}"
    )

    if config["warmup"]:
        config["output_path"] = os.path.join(
            config["output_path"],
            config_data["name"],
            f"{source_name}_to_{target_name}_{method}_{config['output_name']}_warmup",
        )
    else:
        config["output_path"] = os.path.join(
            config["output_path"],
            config_data["name"],
            f"{source_name}_to_{target_name}_{method}_{config['output_name']}",
        )
    if not os.path.exists(config["output_path"]):
        os.system("mkdir -p " + config["output_path"])
    config["out_file"] = open(os.path.join(config["output_path"], "log.txt"), "w")

    ########## MODEL CONFIG ##########
    config_architecture = config["Architecture"]
    ##### BACKBONE CONFIG #####
    backbone_setting = config_architecture["Backbone"]
    ##### CLASSIFIER CONFIG #####
    classifier_setting = config_architecture["Classifier"]

    if backbone_setting["pretrained_1"] and classifier_setting["pretrained_F1"]:
        pass
    
    elif config["pretrained_models"]:
        pretrained_name = f"{source_name}_to_{target_name}_{method}_pretrained_warmup"
        pretrained_path = os.path.join(
            config["pretrained_models"], config_data["name"], pretrained_name
        )
        backbone_setting["pretrained_1"] = (
            f"{pretrained_path}/the_best_G1_pretrained.pth.tar"
        )
        backbone_setting["pretrained_2"] = (
            f"{pretrained_path}/the_best_G2_pretrained.pth.tar"
        )
        classifier_setting["pretrained_F1"] = (
            f"{pretrained_path}/the_best_F1_pretrained.pth.tar"
        )
        classifier_setting["pretrained_F2"] = (
            f"{pretrained_path}/the_best_F2_pretrained.pth.tar"
        )
    ########################################

    # print pout config values
    write_logs(config['out_file'], str(config))

    return config


def write_logs(out_file, log_str, colors=False):
    out_file.write(log_str + "\n")
    out_file.flush()

    if colors:
        print(colored(log_str, color="red", force_color=True))
    else:
        print(log_str)
