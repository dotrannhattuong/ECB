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
    ### Load config ###
    config = load_config(config)
    config["DEVICE"] = torch.device(config["DEVICE"])

    ### set number of classes for each dataset ###
    config_data = config["dataset"]
    config_data["name"] = config_data["data_label"].split("/")[-1].lower()
    if "office31" in config_data["name"]:
        config["Architecture"]["class_num"] = 31
        
    elif "officehome" in config_data["name"]:
        config["Architecture"]["class_num"] = 65
        
    # DomainNet for SSDA and UDA task
    elif "domainnet" in config_data["name"] and config_data["method"] == "SSDA":
        config["Architecture"]["class_num"] = 126
        
    elif "domainnet" in config_data["name"] and config_data["method"] == "UDA":
        config["Architecture"]["class_num"] = 40
        config_data['data_prefix'] = {'train': '_train_mini.txt', 'test': '_test_mini.txt'}
        
    elif "pacs" in config_data["name"]:
        config["Architecture"]["class_num"] = 7
        
    elif "geoimnet" in config_data["name"]:
        config["Architecture"]["class_num"] = 600
        config_data['data_prefix'] = {'train': '_train.txt', 'test': '_test.txt'}

    elif "geoplaces" in config_data["name"]:
        config["Architecture"]["class_num"] = 204
        config_data['data_prefix'] = {'train': '_train.txt', 'test': '_test.txt'}

    elif "visda" in config_data["name"]:
        config["Architecture"]["class_num"] = 12
        
    else:
        raise ValueError(
            "Dataset cannot be recognized. Please define your own dataset here."
        )
    ################################

    ### DATA AUGMENTATION - ASDA ###
    config_data["prep"] = {
        "source_w": preprocess.train_weak(**config_data["prep"]["params"]),
        "source_str": preprocess.source_train_strong(
            **config_data["prep"]["params"]
        ),
        "target_w": preprocess.train_weak(**config_data["prep"]["params"]),
        "target_str": preprocess.target_train_strong(
            **config_data["prep"]["params"]
        ),
        "test": preprocess.image_test(**config_data["prep"]["params"]),
        "val": preprocess.image_val(**config_data["prep"]["params"]),
    }
    ################################

    ### get some infos about task ###
    source_name = config_data["source"]["name"]
    target_name = config_data["target"]["name"]
    target_shot = config_data["target_shot"]
    method = (
        config_data["method"]
        if config_data["method"] == "UDA"
        else f"{config_data['method']}_{target_shot}"
    )
    #################################

    ### Config info for Warmup stage ###
    if config["warmup"]:
        # Folder path for save log.txt and warmup pretrained model
        config["output_path_warmup"] = os.path.join(
            config["pretrained_models"],
            config_data["name"],
            f"{source_name}_to_{target_name}_{method}_pretrained_warmup",
        )
        
        # Create folder if not exist 
        if not os.path.exists(config["output_path_warmup"]):
            os.system("mkdir -p " + config["output_path_warmup"])
        
        # Create File object to write log
        config["out_file_warmup"] = open(os.path.join(config["output_path_warmup"], "log.txt"), "w")
    #####################################
    
    ### Config info for Adaptation stage ###
    # Folder path for save log.txt and adapt pretrained model
    config["output_path"] = os.path.join(
        config["output_path"],
        config_data["name"],
        f"{source_name}_to_{target_name}_{method}_{config['output_name']}",
    )
    
    # Create folder if not exist
    if not os.path.exists(config["output_path"]):
        os.system("mkdir -p " + config["output_path"])
        
    # Create File object to write log
    config["out_file"] = open(os.path.join(config["output_path"], "log.txt"), "w")
    ########################################

    ### Get config model ###
    config_architecture = config["Architecture"]
    # Backbone config   #
    backbone_setting = config_architecture["Backbone"]
    # Classifier config #
    classifier_setting = config_architecture["Classifier"]
    ########################

    ### Set Pretrained model path ###
    # Priority: Warmup Stage (Only use ImageNet pretrain) -> Adapted model path -> Warmuped model path 
    if config["warmup"]:
        backbone_setting["pretrained_1"] = ''
        backbone_setting["pretrained_2"] = ''
        classifier_setting["pretrained_F1"] = ''
        classifier_setting["pretrained_F2"] = ''
        
    elif backbone_setting["pretrained_1"] and classifier_setting["pretrained_F1"]:
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
    #####################################

    return config


def write_logs(out_file, log_str, colors=False):
    out_file.write(log_str + "\n")
    out_file.flush()

    if colors:
        print(colored(log_str, color="red", force_color=True))
    else:
        print(log_str)
