# DATASET

## DomainNet 
* Download **DomainNet** dataset from the [offical website](http://ai.bu.edu/M3SDA/).
* After download, you need to extract the zip file. Your directory tree should look like this:
```bash
domainnet/
│  ├ clipart/
│  ├ infograph/
│  ├ painting/
│  ├ quickdraw/
│  ├ real/
│  ├ sketch/
```

## OfficeHome
* Download **OfficeHome** dataset from the [offical website](http://hemanthdv.org/OfficeHome-Dataset/). Or you can download this dataset using this script.
```bash
pip install -U --no-cache-dir gdown --pre
wget -O OfficeHomeDataset_10072016.zip 'https://docs.google.com/uc?export=download&id=0B81rNlvomiwed0V1YUxQdC1uOTg&confirm=t'
unzip OfficeHomeDataset_10072016.zip
```
* After download, you need to extract the zip file. Your directory tree should look like this:
```bash
office_home/
│  ├ Art/
│  ├ Clipart/
│  ├ Product/
└  └ Real_world/
```

## Office31
* You can use this script to download the **Office31** dataset.
```bash
pip install -U --no-cache-dir gdown --pre
gdown --id 0B4IapRTv9pJ1WGZVd1VDMmhwdlE
tar -xvf domain_adaptation_images.tar.gz
```
* After download and extract the tar file. Your directory tree should look like this:
```bash
office31/
├ amazon/
│  ├ back_pack/
│  ├ bike/
│  └ ⋮
├ dslr/
│  ├ back_pack/
│  ├ bike/
│  └ ⋮
├ webcam/
│  ├ back_pack/
│  ├ bike/
│  └ ⋮
```

## List dataset txts to train SSDA or UDA for these Dataset
```bash
dataset
  #SSDA Dataset
├ domainnet_ssda
│  ├ labeled_source_images_clipart.txt
│  ├ labeled_target_images_clipart_1.txt
│  ├ labeled_target_images_clipart_3.txt
│  ├ unlabeled_target_images_clipart_1.txt
│  ├ unlabeled_target_images_clipart_3.txt
│  ├ validation_target_images_clipart_3.txt
│  └ ⋮
│
├ officehome_ssda
│  ├ labeled_source_images_Art.txt
│  ├ labeled_source_images_Art_1.txt
│  ├ labeled_source_images_Art_3.txt
│  ├ unlabeled_target_images_Art.txt
│  ├ unlabeled_target_images_Art_1.txt
│  ├ unlabeled_target_images_Art_3.txt
│  ├ validation_target_images_Art_3.txt
│  └ ⋮
│
├ office31_ssda
│  ├ labeled_source_images_amazon.txt
│  ├ labeled_source_images_amazon_1.txt
│  ├ labeled_source_images_amazon_3.txt
│  ├ unlabeled_target_images_amazon.txt
│  ├ unlabeled_target_images_amazon_1.txt
│  ├ unlabeled_target_images_amazon_3.txt
│  ├ validation_target_images_amazon_3.txt
│  └ ⋮
│
│ # UDA Dataset
├ domainnet_uda
│  ├ clipart_test_mini.txt
│  ├ clipart_train_mini.txt
│  └ ⋮
│  
├ officehome_uda
│  ├ Art.txt
│  ├ Clipart.txt
│  ├ Product.txt
└  └ Real.txt
```