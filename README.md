# Self-Supervised Video Copy Localization with Regional Token Representation

PyTorch implementation and pretrained models 
for DINOv2. For details, please refer to the original papers.

This paper proposes a self-supervised video copy localization framework. 
The framework incorporates a Regional Token into the Vision Transformer, 
which learns to focus on local regions within each frame using an asymmetric 
training procedure. A novel strategy that leverages the Transitivity 
Property is proposed to generate copied video pairs automatically, which 
facilitates the training of the detector.

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
```
And the expected results are shown below:

| Method          | Seg Recall | Seg Precision | Seg F1 | Video Recall | Video Precision | Video F1 |
|-----------------|------------|---------------|--------|--------------|-----------------|----------|
| Self-Supervised | 69.51      | 68.17         | 68.83  | 91.35        | 99.87           | 95.42    |
| Finetuning      | 75.76      | 67.81         | 71.56  | 93.93        | 99.14           | 96.46    |
