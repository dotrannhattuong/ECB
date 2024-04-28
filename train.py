import argparse
import os
import shutil

import trainer
import trainer_warmup as trainer_warmup 
from log_utils.utils import ReDirectSTD
from model.basenet import build_model
from utils.dataloader import build_data
from utils.utils import build_config, set_seed, write_logs


def parse_opt(known=False):
    parser = argparse.ArgumentParser(description="The proposed method")
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/train.yaml",
        help="config.yaml path",
    )

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args):
    config = build_config(args.cfg)
    set_seed(config["seed"])

    ### write trainer.py ###
    shutil.copy("trainer.py", os.path.join(config["output_path"]))
    ########################

    ### prepare data ###
    config_data = config["dataset"]
    dsets, dset_loaders = build_data(config_data)
    ####################

    ### get model ###    
    config_architecture = config["Architecture"]
    G1, G2, F1, F2 = build_model(config_architecture, DEVICE=config["DEVICE"])
    #################

    ### Warmup Stage ####
    if config["warmup"]:
        
        # Print the config info to the log file
        out_file_warmup = config["out_file_warmup"]
        write_logs(out_file_warmup, str(config))
        
        # Create a new log file for warmup (logs/..._preatrained_warmup.txt)
        log_file_name_warmup = os.path.join("./logs", config_data["name"], config["output_path_warmup"].split("/")[-1] + ".txt")
        re = ReDirectSTD(log_file_name_warmup, "stdout", True)
        
        # Save the trainer_warmup.py file to the output path
        shutil.copy("trainer_warmup.py", os.path.join(config["output_path_warmup"]))
        
        # Begin the warmup stage
        log_str = "==> Step 1: Pre-training on the labeled dataset ..."
        write_logs(out_file_warmup, log_str, colors=True)
        
        # Train on labeled data
        G1, G2, F1, F2 = trainer_warmup.train_labeled_data(
            config, G1, G2, F1, F2, dset_loaders
        )

        log_str = "==> Finished pre-training on source!\n"
        write_logs(out_file_warmup, log_str, colors=True)

        # Save the best model paths when finished
        config_backbone = config["Architecture"]["Backbone"]
        config_backbone["pretrained_1"] = (
            config["output_path_warmup"] + "/the_best_G1_pretrained.pth.tar"
        )
        config_backbone["pretrained_2"] = (
            config["output_path_warmup"] + "/the_best_G2_pretrained.pth.tar"
        )

        config_classifier = config["Architecture"]["Classifier"]
        config_classifier["pretrained_F1"] = (
            config["output_path_warmup"] + "/the_best_F1_pretrained.pth.tar"
        )
        config_classifier["pretrained_F2"] = (
            config["output_path_warmup"] + "/the_best_F2_pretrained.pth.tar"
        )
        
        # Turn off save the content in terminal when finished warmup
        re.set_continue_write(False)
        
        # Load again the model for the next stage
        G1, G2, F1, F2 = build_model(config_architecture, DEVICE=config["DEVICE"])
    ###########################################
    
    ### Adaptation Stage ###    
    # Print the config info to the log file
    out_file = config["out_file"]
    write_logs(out_file, str(config))
    
    # Create a new log file for warmup (logs/..._baseline.txt)
    log_file_name = os.path.join("./logs", config_data["name"], config["output_path"].split("/")[-1] + ".txt")
    ReDirectSTD(log_file_name, "stdout", True, True)
    
    # Begin the adaptation stage
    log_str = "==> Starting the adaptation"
    write_logs(out_file, log_str, colors=True)

    # Train on all data
    G1, G2, F1, F2 = trainer.train(config, G1, G2, F1, F2, dset_loaders)

    log_str = "Finished training and evaluation!"
    write_logs(out_file, log_str, colors=True)
    ##########################


if __name__ == "__main__":
    args = parse_opt()
    main(args)
