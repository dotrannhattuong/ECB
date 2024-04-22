import copy

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model.my_clip import CustomCLIPModel
from model.resnet import resnet34, resnet50


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(7):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x):
        x = self.fc1(x)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


def load_pretrain(model, pretrain_path, device='cpu', init_weight=True):
    if pretrain_path:
        print('Loading:', pretrain_path)
        model.load_state_dict(torch.load(pretrain_path, map_location=device), strict=True)
        print(f'Pretrain weights {model.__class__.__name__} loaded.')
    elif init_weight:
        weights_init(model)
    return model

def build_model(config, DEVICE, pretrain = True):
    config = copy.deepcopy(config)

    ##### BACKBONE CONFIG #####
    backbone_setting = config['Backbone']
    ##### CLASSIFIER CONFIG #####
    classifier_setting = config['Classifier']

    inc_1  = inc_2 = 1000
    ##### BACKBONE 1 #####
    if backbone_setting['name_1'] == 'vit':
        G1 = timm.create_model(model_name='vit_base_patch16_224', pretrained=True)
        
    elif backbone_setting['name_1'] == 'swin':
        G1 = timm.create_model(model_name='swin_base_patch4_window7_224.ms_in1k', pretrained=True)
        
    elif backbone_setting['name_1'] == 'coatnet-0':
        G1 = timm.create_model(model_name='coatnet_0_rw_224.sw_in1k', pretrained=True)
        
    elif backbone_setting['name_1'] == 'coatnet-2':
        G1 = timm.create_model(model_name='coatnet_rmlp_2_rw_224.sw_in1k', pretrained=True)
        
    elif backbone_setting['name_1'] == 'swinv2':
        G1 = timm.create_model(model_name='swinv2_base_window16_256.ms_in1k', pretrained=True)
        
    elif backbone_setting['name_1'] == 'clip':
        G1 = CustomCLIPModel(name = 'ViT-B/32', device=DEVICE)
        inc_1 = 512
        
    elif backbone_setting['name_1'] == 'resnet34':
        G1 = resnet34(pretrained=True, inc=inc_1)
        
    elif backbone_setting['name_1'] == 'resnet50':
        G1 = resnet50(pretrained=True, inc=inc_1)
        
    elif backbone_setting['name_1'] == 'resnet101':
        G1 = models.resnet101(pretrained=True)
        
    elif backbone_setting['name_1'] == "alexnet":
        G1 = AlexNetBase()
        inc_1 = 4096
        
    elif backbone_setting['name_1'] == "vgg":
        G1 = VGGBase()
        inc_1 = 4096
        
    else:
        raise ValueError('Model cannot be recognized.')

    G1 = G1.to(DEVICE)
    ######################

    ##### BACKBONE 2 #####
     
    if backbone_setting['name_2'] == 'vit':
        G2 = timm.create_model(model_name='vit_base_patch16_224', pretrained=True)
    elif backbone_setting['name_2'] == 'swin':
        G2 = timm.create_model(model_name='swin_base_patch4_window7_224.ms_in1k', pretrained=True)

    elif backbone_setting['name_2'] == 'coatnet-0':
        print('G2 co-0')
        G2 = timm.create_model(model_name='coatnet_0_rw_224.sw_in1k', pretrained=True)

    elif backbone_setting['name_2'] == 'coatnet-2': 
        G2 = timm.create_model(model_name='coatnet_rmlp_2_rw_224.sw_in1k ', pretrained=True)

    elif backbone_setting['name_2'] == 'swinv2':
        G2 = timm.create_model(model_name='swinv2_base_window16_256.ms_in1k', pretrained=True)

    elif backbone_setting['name_2'] == 'clip':
        G2 = CustomCLIPModel(name = 'ViT-B/32', device=DEVICE)
        inc_2 = 512    
        
    elif backbone_setting['name_2'] == 'resnet34':
        G2 = resnet34(pretrained=True, inc=inc_2)
        
    elif backbone_setting['name_2'] == 'resnet50':
        G2 = resnet50(pretrained=True, inc=inc_2)
        
    elif backbone_setting['name_2'] == 'resnet101':
        G2 = timm.create_model('resnet101.a1_in1k', pretrained=True)
        
    elif backbone_setting['name_2'] == "alexnet":
        G2 = AlexNetBase()
        inc_2 = 4096
        
    elif backbone_setting['name_2'] == "vgg":
        G2 = VGGBase()
        inc_2 = 4096
    else:
        raise ValueError('Model cannot be recognized.')

    G2 = G2.to(DEVICE)
    ######################

    ##### CLASSIFIER 1 #####
    F1 = Predictor_deep(num_class=config['class_num'], inc=inc_1).to(DEVICE)
    ########################

    ##### CLASSIFIER 2 #####
    F2 = Predictor_deep(num_class=config['class_num'], inc=inc_2).to(DEVICE)
    ########################

    if pretrain:
        ##### LOAD PRETRAIN #####
        G1 = load_pretrain(G1, backbone_setting['pretrained_1'], device=DEVICE, init_weight=False)
        G2 = load_pretrain(G2, backbone_setting['pretrained_2'], device=DEVICE, init_weight=False)
        F1 = load_pretrain(F1, classifier_setting['pretrained_F1'], device=DEVICE)
        F2 = load_pretrain(F2, classifier_setting['pretrained_F2'], device=DEVICE)
        #########################
    return G1, G2, F1, F2