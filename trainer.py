import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.loss import consistency_loss
from utils.optimizer import inv_lr_scheduler
from utils.utils import write_logs


def evaluate(G1, G2, F1, F2, dset_loaders, **kwargs):

    ### Get some variables in kwargs ###
    method = kwargs.get("method", "UDA")
    source_name = kwargs.get("source_name", "clipart")
    target_name = kwargs.get("target_name", "sketch")
    out_file = kwargs.get("out_file", None)
    #####################################
    
    ### Define Log String ###
    log_str = f"  -- Domain task [{source_name} --> {target_name}] \n"

    ### Evaluation ###
    # Evaluate model on Target Test
    target_test_result = eval_domain(
        G1, G2, F1, F2, dset_loaders["target_test"], **kwargs
    )

    source_test_result, target_val_result = {}, {}
    kwargs["return_pseduo"] = False
    
    # Evaluate model on Source Test
    # source_test_result = eval_domain(G1, G2, F1, F2, dset_loaders['source_test'], **kwargs)
    
    # log_str += "CNN's Accuracy Source Test = {:<05.4f}%  ViT's Accuracy Source Test = {:.4f}% \n".format(
    #     source_test_result["cnn_accuracy"], source_test_result["vit_accuracy"]
    # )
    
    # Evaluate model on Target Val in SSDA method
    if method == "SSDA":
        target_val_result = eval_domain(
            G1, G2, F1, F2, dset_loaders["target_val"], **kwargs
        )

        log_str += "\t-- CNN's Accuracy Target Val  = {:<05.4f}%  ViT's Accuracy Target Val =  {:.4f}% \n".format(
            target_val_result["cnn_accuracy"], target_val_result["vit_accuracy"]
        )
    ######################
    
    log_str += "\t-- CNN's Accuracy Target Test = {:<05.4f}%  ViT's Accuracy Target Test = {:.4f}% \n".format(
        target_test_result["cnn_accuracy"], target_test_result["vit_accuracy"]
    )    
    # Write log
    write_logs(out_file, log_str)
    
    ### Set model again to train mode ###
    G1.train()
    G2.train()
    F1.train()
    F2.train()

    return {
        "cnn_acc_source": source_test_result.get("cnn_accuracy", 0.0),
        "vit_acc_source": source_test_result.get("vit_accuracy", 0.0),
        "cnn_acc_target_test": target_test_result.get("cnn_accuracy", 0.0),
        "vit_acc_target_test": target_test_result.get("vit_accuracy", 0.0),
        "cnn_acc_target_val": target_val_result.get("cnn_accuracy", 0.0),
        "vit_acc_target_val": target_val_result.get("vit_accuracy", 0.0),
        "pl_acc_cnn": target_test_result.get("pl_acc_cnn", 0.0),
        "correct_pl_cnn": target_test_result.get("correct_pl_cnn", 0.0),
        "total_pl_cnn": target_test_result.get("total_pl_cnn", 0.0),
        "pl_acc_vit": target_test_result.get("pl_acc_vit", 0.0),
        "correct_pl_vit": target_test_result.get("correct_pl_vit", 0.0),
        "total_pl_vit": target_test_result.get("total_pl_vit", 0.0),
    }


def eval_domain(G1, G2, F1, F2, test_loader, **kwargs):
    ### Get some variables in kwargs ###
    device = kwargs.get("device", "cpu")
    return_pseduo = kwargs.get("return_pseduo", False)
    thresh_cnn = kwargs.get("thresh_cnn", "0.9")
    thresh_vit = kwargs.get("thresh_vit", "0.6")

    ### Set model to eval mode ###
    G1.eval()
    G2.eval()
    F1.eval()
    F2.eval()

    logits_cnn_all, logits_vit_all, labels_all, confidences_cnn_all, confidences_vit_all= [], [], [], [], []
    pl_acc_cnn, correct_pl_cnn, total_pl_cnn, pl_acc_vit, correct_pl_vit, total_pl_vit= 0, 0, 0, 0, 0, 0

    ### Evaluate Stage ###
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs = data["img_1"].to(device)

            # prediction of vit branch
            vit_feature = G1(inputs)
            vit_logits = F1(vit_feature)

            # prediction of cnn branch
            cnn_feature = G2(inputs)
            cnn_logits = F2(cnn_feature)

            # get the logit predictions
            logits_cnn_all.append(cnn_logits.cpu())
            logits_vit_all.append(vit_logits.cpu())
            labels_all.append(data["target"])

            confidences_cnn_all.append(nn.Softmax(dim=1)(logits_cnn_all[-1]).max(1)[0])
            confidences_vit_all.append(nn.Softmax(dim=1)(logits_vit_all[-1]).max(1)[0])
    ######################

    ### Concatenate label ###
    labels = torch.cat(labels_all, dim=0)

    ### Concatenate data ###
    logits_cnn = torch.cat(logits_cnn_all, dim=0)
    logits_vit = torch.cat(logits_vit_all, dim=0)

    ### Predict class labels ###
    _, predict_cnn = torch.max(logits_cnn, 1)
    _, predict_vit = torch.max(logits_vit, 1)

    ### Get accuracy (%) ###
    cnn_accuracy = torch.sum(predict_cnn == labels).item() / len(labels)
    vit_accuracy = torch.sum(predict_vit == labels).item() / len(labels)

    ### Compute Pseudo Label information ###
    if return_pseduo:
        
        def get_pseduo_label_info(confidences_all, predict, labels, thresh):
            confidences = torch.cat(confidences_all, dim=0)
            masks_bool = confidences > thresh
            masks_idx = torch.nonzero(masks_bool, as_tuple=True)[0].numpy()

            # compute accuracy of pseudo labels
            total_pl = len(masks_idx)
            if total_pl > 0:
                correct_pl = torch.sum(predict[masks_bool] == labels[masks_bool]).item()
                pl_acc = correct_pl / total_pl
            else:
                correct_pl = -1.0
                pl_acc = -1.0

            return correct_pl, pl_acc, total_pl

        # Get Info of pseduo label in CNN branch
        correct_pl_cnn, pl_acc_cnn, total_pl_cnn = get_pseduo_label_info(
            confidences_cnn_all, predict_cnn, labels, thresh_cnn
        )
        # Get Info of pseduo label in ViT branch
        correct_pl_vit, pl_acc_vit, total_pl_vit = get_pseduo_label_info(
            confidences_vit_all, predict_vit, labels, thresh_vit
        )

    return {
        "cnn_accuracy": cnn_accuracy * 100,
        "vit_accuracy": vit_accuracy * 100,
        "pl_acc_cnn": pl_acc_cnn,
        "correct_pl_cnn": correct_pl_cnn,
        "total_pl_cnn": total_pl_cnn,
        "pl_acc_vit": pl_acc_vit,
        "correct_pl_vit": correct_pl_vit,
        "total_pl_vit": total_pl_vit,
    }


def train(config, G1, G2, F1, F2, dset_loaders):
    ### Some variables ###
    method = config["dataset"]["method"]
    out_file = config["out_file"]
    config_op = config["optimizer"]
    log_str = f"Method: {method} - File: trainer.py"
    write_logs(out_file, log_str, colors=True)

    DEVICE = config["DEVICE"]
    source_name = config["dataset"]["source"]["name"]
    target_name = config["dataset"]["target"]["name"]
    #######################

    ### Get Param based on base_network ###
    def get_param(base_network, multi=0.1, weight_decay=0.0005):
        param = []
        for key, value in dict(base_network.named_parameters()).items():
            if value.requires_grad:
                if "classifier" not in key:
                    param += [
                        {"params": [value], "lr": multi, "weight_decay": weight_decay}
                    ]
                else:
                    param += [
                        {
                            "params": [value],
                            "lr": multi * 10,
                            "weight_decay": weight_decay,
                        }
                    ]
        return param

    params1 = get_param(G1, multi=0.1, weight_decay=0.0005)
    params2 = get_param(G2, multi=0.1, weight_decay=0.0005)
    #######################################

    ### Set model to train ###
    G1.train()
    G2.train()
    F1.train()
    F2.train()
    ##########################

    ### Set optimizer ###
    optimizer_g1 = optim.SGD(
        params1,
        momentum=config_op["momentum"],
        weight_decay=config_op["weight_decay"],
        nesterov=config_op["nesterov"],
    )
    optimizer_g2 = optim.SGD(
        params2,
        momentum=config_op["momentum"],
        weight_decay=config_op["weight_decay"],
        nesterov=config_op["nesterov"],
    )
    optimizer_f1 = optim.SGD(
        list(F1.parameters()),
        lr=config_op["lr"],
        momentum=config_op["momentum"],
        weight_decay=config_op["weight_decay"],
        nesterov=config_op["nesterov"],
    )
    optimizer_f2 = optim.SGD(
        list(F2.parameters()),
        lr=config_op["lr"],
        momentum=config_op["momentum"],
        weight_decay=config_op["weight_decay"],
        nesterov=config_op["nesterov"],
    )
    #####################

    def zero_grad_all():
        optimizer_g1.zero_grad()
        optimizer_g2.zero_grad()
        optimizer_f1.zero_grad()
        optimizer_f2.zero_grad()

    ### Set loss function ###
    ce_criterion = nn.CrossEntropyLoss().to(DEVICE)

    ### Get Len (total_imgs / batch_size) ###
    len_source_labeled = len(dset_loaders["source_train"])
    len_target_unlabeled = len(dset_loaders["target_train"])
    #########################################

    ### Set best acc ###
    the_best_acc_cnn_source = 0.0
    the_best_acc_vit_source = 0.0
    the_best_acc_cnn_test = 0.0
    the_best_acc_vit_test = 0.0
    the_best_acc_cnn_val = 0.0
    the_best_acc_vit_val = 0.0
    ####################

    ### Check SSDA to define target label ###
    if method == "SSDA":
        len_target_labeled = len(dset_loaders["target_label"])

    ### Set lr ###
    param_lr_g1 = []
    for param_group in optimizer_g1.param_groups:
        param_lr_g1.append(param_group["lr"])
    param_lr_g2 = []
    for param_group in optimizer_g2.param_groups:
        param_lr_g2.append(param_group["lr"])
    param_lr_f1 = []
    for param_group in optimizer_f1.param_groups:
        param_lr_f1.append(param_group["lr"])
    param_lr_f2 = []
    for param_group in optimizer_f2.param_groups:
        param_lr_f2.append(param_group["lr"])
    ###############

    ##### Training #####
    for step in range(config["adapt_iters"]):
        ### Optimizer Scheduler ###
        optimizer_g1 = inv_lr_scheduler(
            param_lr_g1, optimizer_g1, step, init_lr=config_op["lr_vit"]
        )
        optimizer_g2 = inv_lr_scheduler(
            param_lr_g2, optimizer_g2, step, init_lr=config_op["lr_cnn"]
        )
        optimizer_f1 = inv_lr_scheduler(param_lr_f1, optimizer_f1, step, init_lr=config_op["lr_cnn"])
        optimizer_f2 = inv_lr_scheduler(param_lr_f2, optimizer_f2, step, init_lr=config_op["lr_cnn"])

        lr_g1 = optimizer_g1.param_groups[0]["lr"]
        lr_g2 = optimizer_g2.param_groups[0]["lr"]
        ###########################

        ### Load train data ###
        if step % len_source_labeled == 0:
            iter_source = iter(dset_loaders["source_train"])
        if step % len_target_unlabeled == 0:
            iter_unlabeled_target = iter(dset_loaders["target_train"])

        batch_source = next(iter_source)
        batch_target_unlabeled = next(iter_unlabeled_target)
        ########################

        ### Get data ###
        source_w, source_labeled = (
            batch_source["img_1"].to(DEVICE),
            batch_source["target"].to(DEVICE),
        )  # Source Labeled
        inputs_target_w, inputs_target_str = (
            batch_target_unlabeled["img_1"].to(DEVICE),
            batch_target_unlabeled["img_2"].to(DEVICE),
        )  # Target Unlabeled
        ################

        ### Process the Target Label when method is SSDA ###
        if method == "SSDA":
            if step % len_target_labeled == 0:
                iter_labeled_target = iter(dset_loaders["target_label"])

            # Target Labeled
            batch_target_labeled = next(iter_labeled_target)
            target_w, target_labeled = (
                batch_target_labeled["img_1"].to(DEVICE),
                batch_target_labeled["target"].to(DEVICE),
            )

            labeled_targetw_tuple = [source_w, target_w]
            labeled_gt = [source_labeled, target_labeled]

            nl = source_w.size(0) + target_w.size(0)
        #####################################################
        else:  # UDA
            labeled_targetw_tuple = [source_w]
            labeled_gt = [source_labeled]

            nl = source_w.size(0)

        ### Concat Data ###
        labeled_targetw_input = torch.cat(labeled_targetw_tuple + [inputs_target_w], 0)
        labeled_targetstr_input = torch.cat(
            labeled_targetw_tuple + [inputs_target_str], 0
        )
        unlabeled_target_input = torch.cat((inputs_target_w, inputs_target_str), 0)
        labeled_input = torch.cat(labeled_targetw_tuple, 0)
        labeled_gt = torch.cat(labeled_gt, 0)
        #######################

        zero_grad_all()
        ##################################
        ##### I. Supervised Learning #####
        ##################################
        ###### 1. ViT ######
        vit_logits = F1(G1(labeled_input))
        vit_loss = ce_criterion(vit_logits, labeled_gt)

        vit_loss.backward()
        optimizer_g1.step()
        optimizer_f1.step()
        zero_grad_all()

        ###### 2. CNN ######
        cnn_logits = F2(G2(labeled_targetw_input))
        cnn_loss = ce_criterion(cnn_logits[:nl], labeled_gt)

        cnn_loss.backward()
        optimizer_g2.step()
        optimizer_f2.step()
        zero_grad_all()

        #########################
        ######## II. MCD ########
        #########################
        ##### 1. Finding #####
        output_f1 = F1(G1(labeled_input))
        output_f2 = F2(G2(labeled_targetw_input))

        loss_f1 = ce_criterion(output_f1, labeled_gt)
        loss_f2 = ce_criterion(output_f2[:nl], labeled_gt)
        loss_s = loss_f1 + loss_f2

        vit_features_t = G1(unlabeled_target_input)
        output_t1 = F1(vit_features_t)
        output_t2 = F2(vit_features_t)
        loss_dis = torch.mean(
            torch.abs(F.softmax(output_t1, dim=1) - F.softmax(output_t2, dim=1))
        )

        loss = loss_s - loss_dis
        loss.backward()
        optimizer_f1.step()
        optimizer_f2.step()
        zero_grad_all()

        ##### 2. Conquering #####
        for j in range(4):
            feat_t = G2(unlabeled_target_input)
            output_t1 = F1(feat_t)
            output_t2 = F2(feat_t)
            loss_dis = torch.mean(
                torch.abs(F.softmax(output_t1, dim=1) - F.softmax(output_t2, dim=1))
            )

            loss_dis.backward()
            optimizer_g2.step()
            zero_grad_all()

        ############################
        ##### III. Co training #####
        ############################
        ####### 1.ViT -> CNN #######
        vit_logits = F1(G1(inputs_target_w))
        logits_u_s = F2(G2(labeled_targetstr_input))
        loss_vit_to_cnn = consistency_loss(
            logits_u_s[nl:], vit_logits, threshold=config["thresh_ViT"]
        )

        loss_vit_to_cnn.backward()
        optimizer_g2.step()
        optimizer_f2.step()
        zero_grad_all()

        ####### 2.CNN -> ViT #######
        cnn_logits = F2(G2(labeled_targetw_input))
        logits_u_s = F1(G1(inputs_target_str))
        loss_cnn_to_vit = consistency_loss(
            logits_u_s, cnn_logits[nl:], threshold=config["thresh_CNN"]
        )

        loss_cnn_to_vit.backward()
        optimizer_g1.step()
        optimizer_f1.step()
        zero_grad_all()
        ###########################

        ### Print log ###
        if step % 20 == 0 or step == config["adapt_iters"] - 1:
            log_str = (
                "Iters: ({}/{}) \t lr_g1 = {:<10.6f} lr_g2 = {:<10.6f} "
                "CNN's loss = {:<10.6f} ViT's Loss = {:<10.6f} "
                "loss_vit_to_cnn = {:<10.6f} loss_cnn_to_vit = {:<10.6f} \n".format(
                    step,
                    config["adapt_iters"],
                    lr_g1,
                    lr_g2,
                    cnn_loss.item(),
                    vit_loss.item(),
                    loss_vit_to_cnn.item(),
                    loss_cnn_to_vit.item(),
                )
            )

            write_logs(out_file, log_str)

        ##### EVALUATION #####
        if step % config["test_interval"] == config["test_interval"] - 1:
            ### Config for evaluation ###
            eval_kwargs = {
                # for evauation function
                "method": method,
                "device": DEVICE,
                "source_name": source_name,
                "target_name": target_name,
                "out_file": config["out_file"],
                # for eval_domain function
                "return_pseduo": True,
                "thresh_cnn": config["thresh_CNN"],
                "thresh_vit": config["thresh_ViT"],
            }
            eval_result = evaluate(G1, G2, F1, F2, dset_loaders, **eval_kwargs)

            ### Evaluation for Source test ###
            cnn_acc_source = eval_result.get("cnn_acc_source", 0.0)
            vit_acc_source = eval_result.get("vit_acc_source", 0.0)
            if cnn_acc_source > the_best_acc_cnn_source:
                the_best_acc_cnn_source = cnn_acc_source
            if vit_acc_source > the_best_acc_vit_source:
                the_best_acc_vit_source = vit_acc_source

            ### Evaluation for Target val ###
            cnn_acc_val = eval_result.get("cnn_acc_target_val", 0.0)
            vit_acc_val = eval_result.get("vit_acc_target_val", 0.0)
            if cnn_acc_val > the_best_acc_cnn_val:
                the_best_acc_cnn_val = cnn_acc_val
            if vit_acc_val > the_best_acc_vit_val:
                the_best_acc_vit_val = vit_acc_val

            ### Evaluation for Target test ###
            cnn_acc_test = eval_result.get("cnn_acc_target_test", 0.0)
            vit_acc_test = eval_result.get("vit_acc_target_test", 0.0)
            if cnn_acc_test > the_best_acc_cnn_test:
                the_best_acc_cnn_test = cnn_acc_test

                # Save model
                if config["save_models"]:
                    write_logs(out_file, f"  -- Saved CNN Branch (G2 + F2) at {config['output_path']}")
                    torch.save(
                        G2.state_dict(),
                        os.path.join(config["output_path"], "the_best_G2.pth.tar"),
                    )
                    torch.save(
                        F2.state_dict(),
                        os.path.join(config["output_path"], "the_best_F2.pth.tar"),
                    )

            if vit_acc_test > the_best_acc_vit_test:
                the_best_acc_vit_test = vit_acc_test

                # Save model
                if config["save_models"]:
                    write_logs(out_file, f"  -- Saved ViT Branch (G1 + F1) at {config['output_path']}")
                    torch.save(
                        G1.state_dict(),
                        os.path.join(config["output_path"], "the_best_G1.pth.tar"),
                    )
                    torch.save(
                        F1.state_dict(),
                        os.path.join(config["output_path"], "the_best_F1.pth.tar"),
                    )

            # Define log_str to save log
            log_str = f"  -- Domain task [{source_name} --> {target_name}]: \n"
            if the_best_acc_cnn_source != 0.0 or the_best_acc_vit_source != 0.0:
                log_str += "\t-- The best CNN's Acc Source Test = {:<05.4f}% The best Vit's Acc Source Test = {:<05.4f}% \n".format(
                    the_best_acc_cnn_source, the_best_acc_vit_source
                )
            if the_best_acc_cnn_val != 0.0 or the_best_acc_vit_val != 0.0:
                log_str += "\t-- The best CNN's Acc Target Val = {:<05.4f}% The best Vit's Acc Target Val = {:<05.4f}% \n".format(
                    the_best_acc_cnn_val, the_best_acc_vit_val
                )

            log_str += (
                "\t-- The best CNN's Acc Target Test = {:<05.4f}% The best ViT's Acc Target Test = {:<05.4f}% \n"
                "\t-- Acc_Pseudo_Labels_CNN = {:<05.4f} Correct_Pseudo_Labels_CNN = {} Total_Pseudo_Labels_CNN = {:<10} \n"
                "\t-- Acc_Pseudo_Labels_ViT = {:<05.4f} Correct_Pseudo_Labels_ViT = {} Total_Pseudo_Labels_ViT = {:<10} \n".format(
                    the_best_acc_cnn_test,
                    the_best_acc_vit_test,
                    eval_result["pl_acc_cnn"],
                    eval_result["correct_pl_cnn"],
                    eval_result["total_pl_cnn"],
                    eval_result["pl_acc_vit"],
                    eval_result["correct_pl_vit"],
                    eval_result["total_pl_vit"],
                )
            )
            write_logs(out_file, log_str, colors=True)
    return G1, G2, F1, F2