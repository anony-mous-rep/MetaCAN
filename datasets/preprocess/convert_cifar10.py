import os
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

train_file = os.path.join(args.dataset_root, 'cifar10', 'train')
test_file = os.path.join(args.dataset_root, 'cifar10', 'test')

kind_list = os.listdir(train_file)

outlier_train = []
outlier_test = []
normal_train = []
normal_test = []

target_root = './datasets/cifar10_anomaly_detection/cifar10'
test_normal_root = os.path.join(target_root, 'test/good')
if not os.path.exists(test_normal_root):
    os.makedirs(test_normal_root)
test_outlier_root = os.path.join(target_root, 'test/defect')
if not os.path.exists(test_outlier_root):
    os.makedirs(test_outlier_root)
train_root = os.path.join(target_root, 'train/good')
if not os.path.exists(train_root):
    os.makedirs(train_root)
for kind in kind_list:
    print(kind)
    if kind == 'bird' or kind == 'cat' or kind == 'deer' or kind == 'dog' or kind == 'frog' or kind == 'horse':
        normal_train = os.listdir(os.path.join(train_file, kind))
        for f in normal_train:
            source = os.path.join(args.dataset_root, 'cifar10/train/', kind, f)
            shutil.copy(source, train_root +'/'+kind+'_'+f)
        normal_test = os.listdir(os.path.join(test_file, kind))
        for f in normal_test:
            source = os.path.join(args.dataset_root, 'cifar10/test/', kind, f)
            # print(source)
            shutil.copy(source, test_normal_root +'/'+kind+'_'+f)
    else:
        outlier_test = os.listdir(os.path.join(test_file, kind))
        for f in outlier_test:
            source = os.path.join(args.dataset_root, 'cifar10/test/', kind, f)
            # print(source)
            shutil.copy(source, test_outlier_root +'/'+kind+'_'+f)

# 

print('Done')