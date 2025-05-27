
from utils import ImageFilelist
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import mmcv
from os.path import dirname
import pickle
import torchvision
from torchvision import datasets, transforms
import os, json

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'ViT-L/14' #   ViT-B/32 # 'ViT-L/14' 'RN-50' ,'RN-101# 
model, preprocess = clip.load(model_name, device)

# CIFAR-10
out_file = f"outputs/{model_name}_clip_cifar10_image_feature.pkl"
cifar10_test = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
id_image_features = get_image_features(cifar10_test) # numpy features
mmcv.mkdir_or_exist(dirname(out_file))
with open(out_file, 'wb') as f:
        pickle.dump(id_image_features , f)

# CIFAR-100
out_file = f"outputs/{model_name}_clip_cifar100_image_feature.pkl"
cifar100_test = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
id_image_features = get_image_features(cifar100_test) # numpy features
mmcv.mkdir_or_exist(dirname(out_file))
with open(out_file, 'wb') as f:
        pickle.dump(id_image_features , f)

# get all ood datasets for CIFAR100
out_datasets = ['SVHN', 'texture', 'LSUN', 'iSUN', 'places365' ]

for dataset in out_datasets:
    ood_dataset = set_ood_dataset_cifar_100(dataset, preprocess)
    print(len(ood_dataset))
    ood_image_features = get_image_features(ood_dataset)
    ood_out_file = f"outputs/{model_name}_clip_{dataset}_image_feature.pkl"
    mmcv.mkdir_or_exist(dirname(ood_out_file))
    with open(ood_out_file, 'wb') as f:
        pickle.dump(ood_image_features , f)
