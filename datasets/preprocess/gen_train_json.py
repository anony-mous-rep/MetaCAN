import os
import glob
import json
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('AD dataset json generation', add_help=False)
    parser.add_argument('--dataset_dir', type=str, help='path to dataset dir',
                        default='./datasets/cifar10_anomaly_detection')
    parser.add_argument('--output_dir', type=str, help='path to output dir',
                        default='./datasets/AD_json/JSON_PATH')
    parser.add_argument('--dataset_name', type=str, help='dataset name',
                        default='cifar10')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    dataset_list = os.listdir(args.dataset_dir)
    # for dataset_name in dataset_list:
    # args.dataset_name = dataset_name
    image_dir = os.path.join(args.dataset_dir, args.dataset_name)
    save_path_normal = os.path.join(args.output_dir, args.dataset_name, args.dataset_name+"_train_normal.json")
    save_path_outlier = os.path.join(args.output_dir,  args.dataset_name, args.dataset_name + "_train_outlier.json")
    if not os.path.exists(os.path.join(args.output_dir, args.dataset_name)):
        os.makedirs(os.path.join(args.output_dir, args.dataset_name))
    print(save_path_normal, save_path_outlier)

    normal_data = list()
    normal_output_dict = list()
    normal_train_files = os.listdir(os.path.join(image_dir, 'train', 'good'))
    # print(normal_train_files)
    for file in normal_train_files:
        path_dict = {}
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'JPG' in file[-3:] or 'jpeg' in file[-4:]:
            normal_data.append('train' + '/good/' + file)
            path_dict["image_path"] = os.path.join(image_dir, 'train', 'good', file)
            path_dict["target"] = 0
            path_dict["type"] = args.dataset_name
            normal_output_dict.append(path_dict)
    normal_test_files = os.listdir(os.path.join(image_dir, 'test', 'good'))
    for file in normal_test_files:
        path_dict = {}
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'JPG' in file[-3:] or 'jpeg' in file[-4:]:
            normal_data.append('test' + '/good/' + file)
            path_dict["image_path"] = os.path.join(image_dir, 'test', 'good', file)
            path_dict["target"] = 0
            path_dict["type"] = args.dataset_name
            normal_output_dict.append(path_dict)

    outlier_output_dict = list()
    outlier_data = list()
    outlier_data_dir = os.path.join(image_dir, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl in outlier_classes:
        if cl == 'good':
            continue
        outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
        for file in outlier_file:
            path_dict = {}
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'JPG' in file[-3:] or 'jpeg' in file[-4:]:
                outlier_data.append('test/' + cl + '/' + file)
                path_dict["image_path"] = os.path.join(image_dir, 'test', cl, file)
                path_dict["target"] = 1
                path_dict["type"] = args.dataset_name
                outlier_output_dict.append(path_dict)

    json.dump(normal_output_dict, open(save_path_normal, 'w'))
    json.dump(outlier_output_dict, open(save_path_outlier, 'w'))
