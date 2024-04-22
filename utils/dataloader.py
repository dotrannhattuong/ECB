import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageList(Dataset):
    def __init__(
        self,
        data_root,
        data_path,
        transform_w=None,
        transform_str=None,
        sample_masks=None,
        use_cgct_mask=False,
    ):
        self.data_root = data_root
        self.data_path = data_path  # name of the domain

        self.transform_w = transform_w
        self.transform_str = transform_str

        self.loader = self._rgb_loader
        self.sample_masks = sample_masks
        self.use_cgct_mask = use_cgct_mask

        self.imgs = self._make_dataset(data_path)

        if self.use_cgct_mask:
            self.sample_masks = (
                sample_masks
                if sample_masks is not None
                else torch.zeros(len(self.imgs)).float()
            )
        else:
            if sample_masks is not None:
                temp_list = self.imgs
                self.imgs = [temp_list[i] for i in self.sample_masks]

    def _rgb_loader(self, path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB")

    def _make_dataset(self, image_list_path):
        image_list = open(image_list_path).readlines()
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
        return images

    def __getitem__(self, index):
        output = {}
        path, target = self.imgs[index]

        img = self.loader(os.path.join(self.data_root, path))

        if self.transform_w is not None:
            img_1 = self.transform_w(img)
        if self.transform_str is not None:
            img_2 = self.transform_str(img)
            output["img_2"] = img_2

        output["img_1"] = img_1

        output["target"] = torch.squeeze(torch.LongTensor([np.int64(target).item()]))

        output["idx"] = index
        if self.use_cgct_mask:
            output["mask"] = torch.squeeze(
                torch.LongTensor([np.int64(self.sample_masks[index]).item()])
            )

        return output

    def __len__(self):
        return len(self.imgs)


def build_data(config, debug=True):
    dsets = {}
    dset_loaders = {}

    source_bs = config["source"]["batch_size"]
    target_bs = config["target"]["batch_size"]

    source_name = config["source"]["name"]
    target_name = config["target"]["name"]

    method = config["method"]
    data_root = config["data_root"]
    data_label = config["data_label"]

    if method == "SSDA":
        num = config["target_shot"]

        image_set_file_s = os.path.join(data_label, 'labeled_source_images_' + source_name + '.txt')
        image_set_file_s_test = os.path.join(data_label, 'validation_target_images_' + source_name + '_3.txt')

        image_set_file_tl = os.path.join(data_label, 'labeled_target_images_' + target_name + '_%d.txt' % (num))

        image_set_file_tu = image_set_file_tu_test = os.path.join(data_label, 'unlabeled_target_images_' + target_name + '_%d.txt' % (num))
        image_set_file_tu_val = os.path.join(data_label, 'validation_target_images_' + target_name + '_3.txt')

        ############## TARGET LABEL DATA SET ##############
        dsets["target_label"] = ImageList(
            data_root=data_root,
            data_path=image_set_file_tl,
            transform_w=config["prep"]["val"],
            transform_str=config["prep"]["target_str"],
        )

        ############## TARGET LABEL DATA LOADER ##############
        dset_loaders["target_label"] = DataLoader(
            dsets["target_label"],
            batch_size=target_bs,
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=True,
            pin_memory=True,
        )

        ############## TARGET UNLABEL VAL DATA SET ##############
        dsets["target_val"] = ImageList(
            data_root=data_root,
            data_path=image_set_file_tu_val,
            transform_w=config["prep"]["val"],
            use_cgct_mask=config["use_cgct_mask"],
        )

        ############## TARGET UNLABEL VAL DATA LOADER ##############
        dset_loaders["target_val"] = DataLoader(
            dataset=dsets["target_val"],
            batch_size=target_bs,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

    elif method == "UDA":
        prefix = config.get("data_prefix", {'train':'.txt', 'test': '.txt'})
        image_set_file_s = os.path.join(data_label, source_name + prefix['train'])
        image_set_file_s_test = os.path.join(data_label, source_name + prefix['test'])

        image_set_file_tu = os.path.join(data_label, target_name + prefix['train'])
        image_set_file_tu_test = os.path.join(data_label, target_name + prefix['test'])

    if debug:
        # debug here
        print("="*10, 'DATA PATH', "="*10)
        print(f"source train: {image_set_file_s}")
        print(f"source test: {image_set_file_s_test}")
        print(f"target train: {image_set_file_tu}")
        print(f"target test: {image_set_file_tu_test}")
        print("="*31)

    ############## SOURCE DATA SET ##############
    dsets["source_train"] = ImageList(
        data_root=data_root,
        data_path=image_set_file_s,
        transform_w=config["prep"]["source_w"],
        transform_str=config["prep"]["source_str"],
    )

    dsets["source_test"] = ImageList(
        data_root=data_root,
        data_path=image_set_file_s_test,
        transform_w=config["prep"]["test"],
        use_cgct_mask=config["use_cgct_mask"],
    )

    ############## SOURCE DATA LOADER ##############
    dset_loaders["source_train"] = DataLoader(
        dsets["source_train"],
        batch_size=source_bs,
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    dset_loaders["source_test"] = DataLoader(
        dsets["source_test"],
        batch_size=source_bs,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    ############## TARGET UNLABEL DATA SET ##############
    dsets["target_train"] = ImageList(
        data_root=data_root,
        data_path=image_set_file_tu,
        transform_w=config["prep"]["target_w"],
        transform_str=config["prep"]["target_str"],
        use_cgct_mask=config["use_cgct_mask"],
    )

    dsets["target_test"] = ImageList(
        data_root=data_root,
        data_path=image_set_file_tu_test,
        transform_w=config["prep"]["test"],
        use_cgct_mask=config["use_cgct_mask"],
    )

    ############## TARGET UNLABEL DATA LOADER ##############
    dset_loaders["target_train"] = DataLoader(
        dataset=dsets["target_train"],
        batch_size=target_bs,
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    dset_loaders["target_test"] = DataLoader(
        dataset=dsets["target_test"],
        batch_size=target_bs,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    # ==============================================================

    return dsets, dset_loaders
