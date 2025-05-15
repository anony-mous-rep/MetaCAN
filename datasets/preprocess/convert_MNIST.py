import os
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

train_file = os.path.join(args.dataset_root, 'mnist_png', 'training')
test_file = os.path.join(args.dataset_root, 'mnist_png', 'testing')

kind_list = os.listdir(train_file)

outlier_train = []
outlier_test = []
normal_train = []
normal_test = []

target_root = './datasets/mnist_anomaly_detection/mnist'
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
    if kind == '0' or kind == '2' or kind == '4' or kind == '6' or kind == '8':
        normal_train = os.listdir(os.path.join(train_file, kind))
        for f in normal_train:
            source = os.path.join(args.dataset_root, 'mnist_png/training/', kind, f)
            shutil.copy(source, train_root)
        normal_test = os.listdir(os.path.join(test_file, kind))
        for f in normal_test:
            source = os.path.join(args.dataset_root, 'mnist_png/testing/', kind, f)
            # print(source)
            shutil.copy(source, test_normal_root)
    else:
        outlier_test = os.listdir(os.path.join(test_file, kind))
        for f in outlier_test:
            source = os.path.join(args.dataset_root, 'mnist_png/testing/', kind, f)
            # print(source)
            shutil.copy(source, test_outlier_root)

# 

print('Done')