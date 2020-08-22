## More Classifiers, Less Forgetting: A Generic Multi-classifier Paradigm for Incremental Learning (ECCV 2020)
- Exploit the classifier ensemble for reducing forgetting on learning tasks incrementally.
- Extend two regularization methods (MAS and LwF) focusing on parameter and activation regularization.
- Obtain consistent improvements over the single-classifier paradigm.

![architecture](https://github.com/Liuy8/MUC/blob/master/MUC_overview.png)

## Dependencies

- PyTorch 
- Python 
- Numpy
- scipy

## Data

- Download the dataset (CIFAR-100, Tiny-ImageNet, SVHN) and save them to the 'data' directory.
- SVHN is used as an out-of-distribution dataset for training additional side classifiers.


## Experiment on CIFAR-100 incremental benchmark

- Run ```cifar100_MUC_MAS.py``` to train the MUC-MAS method.

- Run ```cifar100_MUC_LwF.py``` to train the MUC-LwF method.

## Experiment on Tiny-ImageNet incremental benchmark

- Run ```tinyimagenet_MUC_MAS.py``` to train the MUC-MAS method.

- Run ```tinyimagenet_MUC_LwF.py``` to train the MUC-LwF method.

## Notes
- Some codes are based on the codebase of the [repository](https://github.com/hshustc/CVPR19_Incremental_Learning).
- More instructions will be provided later.

# Citation
Please cite the following paper if it is helpful for your research:
```
@InProceedings{MUC_ECCV2020,
author = {Liu, Yu and Parisot, Sarah and Slabaugh, Gregory and Jia, Xu and Leonardis,Ales and Tuytelaars, Tinne}
title = {More Classifiers, Less Forgetting: A Generic Multi-classifier Paradigm for Incremental Learning},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020}
}
```
