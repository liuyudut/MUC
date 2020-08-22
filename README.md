# More Classifiers, Less Forgetting: A Generic Multi-classifier Paradigm for Incremental Learning (ECCV 2020)
- Exploit the classifier ensemble for reducing forgetting on learning tasks incrementally.
- Extend two regularization methods (MAS and LwF) focusing on parameter and activation regularization.
- Obtain consistent improvements over the single-classifier paradigm.

![architecture](https://github.com/Liuy8/Explainable-ZSL/blob/master/diversity_consistency.png)

## Dependencies

- PyTorch 
- Python 
- Numpy
- scipy

## Data

- Download the dataset (CIFAR-100, Tiny-ImageNet, SVHN) and save them to the 'data' directory.
- SVHN is used as an out-of-distribution dataaset to train additional side classifiers.


## Experiment on CIFAR-100 incremental benchmark

- Run ```cifar100_MUC_MAS.py``` to train the MUC-MAS method.

- Run ```cifar100_MUC_LwF.py``` to train the MUC-LwF method.

## Experiment on Tiny-ImageNet incremental benchmark

- Run ```tinyimagenet_MUC_MAS.py``` to train the MUC-MAS method.

- Run ```tinyimagenet_MUC_LwF.py``` to train the MUC-LwF method.

## Notes

More instructions will be provided later.
