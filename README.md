
## Device 
A100 80g

## How to Run
## 1.pip packages
- pip install -r requirements.txt
- 
## 2.Download Public Dataset.Because these datasets are from other papers, we don't provide url, you can download by yourself. (VisA, ELPV, mnist and so on).

The dataset folder structure should look like:
```
DATA_PATH/
    subset_1/
        train/
            good/
        test/
            good/
            defect_class_1/
            defect_class_2/
            defect_class_3/
            ...
    ...
```

## 3.Generate Training/Test Json Files.

The json folder structure should look like:
```
JSON_PATH/
    dataset_1/
        subset_1/
            subset_1_train_normal.json
            subset_1_train_outlier.json
            subset_1_val_normal.json
            subset_1_val_outlier.json
        subset_2/
        subset_3/
        ...
    ...
```

## 4. Download the normal refer_samples by anonymous URL(https://drive.google.com/drive/folders/1WV34-1DupqmT06Til6iNA-YqINyBzIhW?usp=sharing)
put refer_samples in path './refer_samples'

## 4. Download the Train_on_mvtec Model Checkpoints on anonymous URL(https://drive.google.com/file/d/1ppamboj4kI5q-UsSLWO4OgYn8s_rxffh/view)
put checkpoints in path './checkpoints/trained_on_mvtec/checkpoint_10.pyth'

## 5. Test
```bash
python test.py --val_normal_json_path $normal-json-files-for-testing --val_outlier_json_path $abnormal-json-files-for-testing --category $dataset-class-name --few_shot_dir $path-to-few-shot-samples
```

For example, if run on the category `candle` of `visa` with `k=2`:
```bash
python test.py --val_normal_json_path /JSON_PATH/visa/candle_val_normal.json --val_outlier_json_path /JSON_PATH/visa/candle_val_outlier.json --category candle --few_shot_dir /refer_samples/visa/2/
```

