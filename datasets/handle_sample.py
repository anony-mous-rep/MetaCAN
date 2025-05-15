import os
from PIL import Image

if __name__ == '__main__':
    kind_list = os.listdir('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec-900')
    for kind in kind_list:
        train_good_list = os.listdir(os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec-900', kind, 'train/good'))
        for train_good_img in train_good_list:
            image = Image.open(os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec-900', kind, 'train/good', train_good_img))
            image_resized = image.resize((240,240), Image.BICUBIC)
            save_name = os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec', kind, 'train/good', train_good_img)
            if not os.path.exists(os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec', kind, 'train/good')):
                os.makedirs(os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec', kind, 'train/good'))
            image_resized.save(save_name)
        test_kind_list = os.listdir(os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec-900', kind, 'test'))
        for test_kind in test_kind_list:
            test_image_list = os.listdir(os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec-900', kind, 'test', test_kind))
            for test_image in test_image_list:
                image = Image.open(
                    os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec-900', kind, 'test', test_kind, test_image))
                image_resized = image.resize((240, 240), Image.BICUBIC)
                save_name = os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec', kind, 'test', test_kind, test_image)
                if not os.path.exists(os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec', kind, 'test', test_kind)):
                    os.makedirs(os.path.join('/mnt/nvme0n1/code/InCTRL/datasets/AD_json/mvtec', kind, 'test', test_kind))
                image_resized.save(save_name)



