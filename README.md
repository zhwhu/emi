# emi
#This repo is heavily based on (https://github.com/gydpku/OCM), many thanks.
Running Experiments
Environments:
	Pytorch1.6
	Numpy
	Scipy
	Apex
  timm
Dataset:CIFAR10
    python emi_cifar10.py --buffer_size 1000
Dataset:CIFAR100
    python emi_cifar100.py --buffer_size 1000
Dataset:TinyImageNet
    python emi_imagenet.py --buffer_size 2000
