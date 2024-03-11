# Self-Supervised Video Copy Localization with Regional Token Representation

PyTorch implementation and pretrained models 
for RTR. For details, please refer to the original papers.

## Preparation

### Environment
```shell
pip install -r requirements.txt
```

### Features
We upload the regional token representation to 
[Google Drive](https://drive.google.com/file/d/1Q25tt80ekLxUgukXvukB0jMUZtdUCcbe/view?usp=drive_link), 
which was registered only for open-source anonymously. 
To prepare `features` folder,
1. Download it and put it in root of project folder
2. Unzip it to `features` folder
```shell
tar -xzvf features.tar.gz
```
3. Delete the zip file
```shell
rm features.tar.gz
```

### Models

We upload the checkpoint of training model to 
[Google Drive](https://drive.google.com/file/d/1DhGdbRogXR1C9lgcrrEU73tJBkZj5k8j/view?usp=drive_link)
as well. To prepare `weights` folder,
1. Download it and put it in root of project folder
2. Unzip it to `weights` folder
```shell
tar -xzvf weights.tar.gz
```
3. Delete the zip file
```shell
rm weights.tar.gz
```

## Run
```shell
bash test_eval.sh [option1] [option2] [option3]
[option1] - CKPT: path of model checkpoint.
[option2] - DATASET: 'VCDB' or 'VCSL'
[option3] - FEAT: 'RTR'
```

To reproduce the  results of our RTR model on VCSL dataset, please run
```shell
# self-supervised on VCSL
bash test_eval.sh weights/ours_ssl_rtr.pth VCSL RTR

# finetuning on VCSL
bash test_eval.sh weights/ours_ft_rtr.pth VCSL RTR

# self-supervised on VCDB
bash test_eval.sh weights/ours_ssl_rtr.pth VCDB RTR

# finetuning on VCDB
bash test_eval.sh weights/ours_ft_rtr.pth VCDB RTR
```
And the expected results are shown below:

| Dataset | Method    | Seg Recall | Seg Precision | Seg F1 | Video Recall | Video Precision | Video F1 |
|---------|-----------|------------|---------------|--------|--------------|-----------------|----------|
| VCSL    | Ours-ssl  | 69.51      | 68.17         | 68.83  | 91.35        | 99.87           | 95.42    |
| VCSL    | Ours-ft   | 75.76      | 67.81         | 71.56  | 93.93        | 99.14           | 96.46    |
| VCDB    | Ours-ssl  | 78.98      | 75.61         | 77.26  | 87.46        | 100.00          | 93.31    |
| VCDB    | Ours-ft   | 80.74      | 76.91         | 78.78  | 88.89        | 100.00          | 94.12    |
