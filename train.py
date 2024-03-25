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
    #########################

    ### prepare data ###
    config_data = config["dataset"]
    log_file_name = os.path.join(
        "./logs", config_data["name"], config["output_path"].split("/")[-1] + ".txt"
    )
    ReDirectSTD(log_file_name, "stdout", True)  # Check

    dsets, dset_loaders = build_data(config_data)
    ####################

    ### get model ###
    out_file = config['out_file']
    config_architecture = config["Architecture"]
    G1, G2, F1, F2 = build_model(config_architecture, DEVICE=config["DEVICE"])
    #################

    #################
    ### Training ####
    #################
    # train on labeled data
    if config["warmup"]:
        log_str = "==> Step 1: Pre-training on the labeled dataset ..."
        write_logs(out_file, log_str, colors=True)

        G1, G2, F1, F2 = trainer_warmup.train_labeled_data(
            config, G1, G2, F1, F2, dset_loaders
        )

        log_str = "==> Finished pre-training on source!\n"
        write_logs(out_file, log_str, colors=True)

        # Load best weights
        config_backbone = config["Architecture"]["Backbone"]
        config_backbone["pretrained_1"] = (
            config["output_path"] + "/the_best_G1_pretrained.pth.tar"
        )
        config_backbone["pretrained_2"] = (
            config["output_path"] + "/the_best_G2_pretrained.pth.tar"
        )

        config_classifier = config["Architecture"]["Classifier"]
        config_classifier["pretrained_F1"] = (
            config["output_path"] + "/the_best_classifier_1_pretrained.pth.tar"
        )
        config_classifier["pretrained_F2"] = (
            config["output_path"] + "/the_best_classifier_2_pretrained.pth.tar"
        )

        G1, G2, F1, F2 = build_model(config_architecture, DEVICE=config["DEVICE"])

    # run adaptation episodes
    log_str = "==> Starting the adaptation"
    write_logs(out_file, log_str, colors=True)

    G1, G2, F1, F2 = trainer.train(config, G1, G2, F1, F2, dset_loaders)

    log_str = "Finished training and evaluation!"
    write_logs(out_file, log_str, colors=True)


if __name__ == "__main__":
    args = parse_opt()
    main(args)
