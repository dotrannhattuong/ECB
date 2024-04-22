# CONFIG
**NOTE**:
```bash
* Priorities paths: warmup stage -> weights with ECB method -> pretrained_models.
* If you want to train warmup: Set `warmup`: True.
* Don't use any pretrained model: Set `pretrained_models` and pretrained backbone paths are blank.
* Only use warmup pretrained model: Set `pretrained_models` is the folder contains the warmup pretrained model paths.
* Use pretrained model that train with ECB method: Set four pretrained model paths. 
```

* Folowing is the config arguments and description in the config file.
``` bash
Config
│  ├ DEVICE: cuda:1 # Device GPU ID (cpu for CPU Implementation)
│  ├ ndomains: 2 # Number domains
│  ├ output_path: ./results # Output folder path to save the checkpoint when training
│  ├ output_name: baseline # Output folder name when save the models
│  ├ source_iters: 100000 # Number iter in supervised training stage.
│  ├ adapt_iters: 50000 # Number iter in adaptation training stage.
│  ├ test_interval: 500 # Evaluatate the model when iter step equal test_interval
│  ├ seed: 1 # Controls randomness, making results repeatable for debugging and comparisons.
│  ├ warmup: False # The Flag to train labeled samples (If True)
│  ├ pretrained_models: ./pretrained_models # The Folder contains the warmup pretrained models
│  ├ lamda: 0.1 # value of lamda for Domain Adaptation (adentropy loss)
│  ├ save_models: True # Save models when training or Not
│  ├ thresh_ViT: 0.6 # The threshold for ViT branch when calculate Pseduo Label
│  ├ thresh_CNN: 0.9 # The threshold for CNN branch when calculate Pseduo Label
│  │ 
│  ├ dataset:
│  │  ├ method: SSDA # UDA or SSDA method
│  │  ├ data_root: ../Dataset/DomainNet # Dataset image folder path
│  │  ├ data_label: ./dataset/domainnet_ssda # Dataset txt folder path
│  │  ├ num_workers: 4 # Number of workders for dataloader
│  │  ├ target_shot: 3 # Number of target for train SSDA setting. SSDA: 1 - 3 - UDA: 0
│  │  ├ use_cgct_mask: True # Used for in ImageList (torch.dataset)
│  │  │
│  │  ├ source: 
│  │  │  ├ name: real # Source domain name
│  │  │  └ batch_size: 32 # Source batch size
│  │  ├ target:
│  │  │  ├ name: clipart # Target domain name
│  │  │  └ batch_size: 32 # Target batch size
│  │  │
│  │  ├ prep: 
│  │  │  ├ params:
│  │  │  │  ├ resize_size: 256 # Resize image in augment
│  │  └  └  └ crop_size: 224 # Resize crop image in augment
│  │    
│  ├ optimizer:
│  │  ├ lr: 1.0 # Learning rate for Classifier F1 and F2
│  │  ├ lr_vit: 0.001 # Learning rate for Vit branch using in scheduler function
│  │  ├ lr_cnn: 0.01 # Learning rate for CNN branch using in scheduler function
│  │  ├ momentum: 0.9 # Momentum parameter for SGD Optimizer
│  │  ├ weight_decay: 0.0005 # Weight_decay parameter
│  │  └ nesterov: True # Use nesterov SGD or not
│  │  
│  ├ Architecture:
│  │  ├ Backbone: 
│  │  │  ├ name_1: vit # The model name for ViT Branch 
│  │  │  ├ pretrained_1: # The pretrained model path for ViT
│  │  │  ├ name_2: resnet34 # The model name for CNN Branch 
│  │  │  └ pretrained_2: # The pretrained model path for CNN
│  │  │
│  │  ├ Classifier:
│  │  │  ├ name_1: MLP # The model name for Classifier F1
│  │  │  ├ pretrained_F1: # The pretrained model path for F1
│  │  │  ├ name_2: MLP # The model name for Classifier F2
└  └  └  └ pretrained_F2: # The pretrained model path for F2  
```