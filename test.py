import argparse

import trainer as trainer
from model.basenet import build_model
from utils.dataloader import build_data
from utils.utils import build_config, set_seed, write_logs
import os
from log_utils.utils import ReDirectSTD


def parse_opt(known=False):
    parser = argparse.ArgumentParser(description="The proposed method")
    parser.add_argument(
        "--cfg", type=str, default="configs/test.yaml", help="model.yaml path"
    )

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args):
    ### Load the config file ###
    config = build_config(args.cfg)
    out_file = config['out_file']
    write_logs(out_file, str(config))
    
    DEVICE = config["DEVICE"]
    set_seed(config["seed"])
    ############################

    config_data = config["dataset"]
    
    ### Create a new log file for warmup (logs/..._test.txt) ###
    log_file_name = os.path.join(
        "./logs", config_data["name"], config["output_path"].split("/")[-1] + ".txt"
    )
    ReDirectSTD(log_file_name, "stdout", True) 
    ############################################################
    
    ### Get config ###
    # Get data config
    dsets, dset_loaders = build_data(config_data)
    source_name = config["dataset"]["source"]["name"]
    target_name = config["dataset"]["target"]["name"]

    # get model config
    config_architecture = config["Architecture"]
    G1, G2, F1, F2 = build_model(config_architecture, DEVICE=DEVICE)
    ##################

    ### Testing stage ####    
    eval_kwargs = {
        # for evauation function
        "method": config_data["method"],
        "device": DEVICE,
        "out_file": config["out_file"],
        # for eval_domain function
        "return_pseduo": True,
        "thresh_cnn": config["thresh_CNN"],
        "thresh_vit": config["thresh_ViT"],
    }

    # Evaluate the model on the target test set
    test_target_res = trainer.eval_domain(
        G1, G2, F1, F2, dset_loaders["target_test"], **eval_kwargs
    )

    local_acc_target_test, global_acc_target_test = (
        test_target_res["cnn_accuracy"],
        test_target_res["vit_accuracy"],
    )

    # Define the log string
    log_str = (
        "\n============ TESTING ============"
        "\n-- Domain task [{} --> {}]: "
        "\n  -- The best CNN's Acc Target Test = {:<05.4f}% The best ViT's Acc Target Test = {:<05.4f}% \n".format(
            source_name, target_name, local_acc_target_test, global_acc_target_test
        )
    )
    
    # Write the log string to the log file
    write_logs(out_file, log_str, colors=True)
    ######################


if __name__ == "__main__":
    args = parse_opt()
    main(args)
