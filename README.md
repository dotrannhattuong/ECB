# Learning CNN on ViT: A Hybrid Model to Explicitly Class-specific Boundaries for Domain Adaptation
This repository contains the code of the ECB method for Classification in Domain Adaptation.

> Ba-Hung Ngo*, Nhat-Tuong Do-Tran*, Tuan-Ngoc Nguyen, Hae-Gon Jeon and Tae Jong Choi† 
<br>Accepted In IEEE/CVF Conference on Computer Vision and Pattern Recognition (<a href="https://cvpr.thecvf.com/Conferences/2024/">CVPR 2024</a>).

<a href="/">Paper</a> | 
<a href="/">Project Page</a> | 

## Proposed Method 
<br/>
<!-- <div style="display:flex; background: #fff; align-item:center;">
<figure id="method" style="margin-right:0px">
    <img src="./images/method_1.svg" />
</figure>
<figure id="method_2">
<img src="./images/method_2.svg" />
</figure> -->

<figure id="method" style="background: #fff; padding:10px; margin:0px">
    <img src="./images/method_1.svg" style=""/>
    
</figure>
<br/>

* <b><i>Supervised Training:</i></b> We train both ViT and CNN branches on labeled samples.
* <b><i>Finding To Conquering Strategy (FTC):</i></b> We find class-specific boundaries based on the fixed ViT Encoder E1 by maximizing discrepancy between the Classifier F1 and F2. Subsequently, the CNN Encoder E2 clusters the target features based on those class-specific boundaris by minimizing discrepancy.
<!-- * <b><i>Co-training Training: </i></b> We apply co-training to exchange effectively knowledge between two branches on unlabeled samples , the ViT branch E1 generates a pseudo label for weakly unlabeled samples to teach the CNN branch E2 with strongly unlabeled samples. -->

## Prepare
### Dataset
Please follow the instructions in [https://github.com/dotrannhattuong/ECB/dataset/DATASET.md](DATASET.md) to download datasets.

### Installation
```bash
conda env create -f environment.yml
```

## Training
* The train.yaml is the config file for training our method. You can change the arguments to train Semi-Supervised Domain Adaptation (SSDA) or Unsupervised Domain Adaptation (UDA).
```bash
python train.py --cfg configs/train.yaml
```
## Evaluation
* If you need evaluate the test dataset with our pretrained model. You need to download these checkpoint.
```bash
sh download_pretrain.sh
```

* For evaluation, you need to modify the configuration arguments in test/yaml in the configs folder. These arguments are described in [https://github.com/dotrannhattuong/ECB/configs/CONFIG.md](CONFIG.md)
```bash
python test.py --cfg configs/test.yaml
```

## Visualization
* The visualization compares features from two networks (CNN, ViT) for the <b><i>real --> sketch</i></b>  on the <b>DomainNet</b> dataset in the 3-shot scenario, before and after adaptation with the FTC strategy.
<div class="figure*" style="display:flex; background: #fff; align-item:center;">
    <figure id="visualize_tsne_a" style="padding:2px;margin:0px;">
        <img src="./images/cnn_before.svg" />
        <p style='color: #000; text-align:center; line-height:1px; padding-top: 10px;' > (a) </p>
    </figure>   
    <figure id="visualize_tsne_b" style="padding:2px;margin:0px;">
        <img src="./images/vit_before.svg" />
        <p style='color: #000; text-align:center; line-height:1px; padding-top: 10px;' > (b) </p>
    </figure>
    <figure id="visualize_tsne_c" style="padding:2px;margin:0px;">
        <img src="./images/cnn_wo_FTC.svg" />
        <p style='color: #000; text-align:center; line-height:1px; padding-top: 10px;' > (c) </p>
    </figure>
    <figure id="visualize_tsne_d" style="padding:2px;margin:0px;">
        <img src="./images/cnn.svg" />
        <p style='color: #000; text-align:center; line-height:1px; padding-top: 10px;' > (d) </p>
    </figure>
</div>
<br />

* The visualization in a few samples using GRAD-CAM technique to show to performance for CNN and ViT when applying ECB method.
<figure id="gradcam" style="background: #fff; margin:0px; text-align: center; padding:10px 0px">
    <img src="./images/gradcam.png" />
</figure>

## Citation
```
@inproceedings{,
  title={Learning CNN on ViT: A Hybrid Model to Explicitly Class-specific Boundaries for Domain Adaptation},
  author={},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```