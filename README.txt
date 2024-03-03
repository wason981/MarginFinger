# MarginFinger

Implementation of the papaer titled "MarginFinger: Controlling Generated Fingerprint Distance to Classification boundary Using Conditional GANs"

![Framework](https://github.com/wason981/MarginFinger/blob/main/images/framework.png)

### Run the Code

#### 0. Requirements

The package using in this project was listed in requirements.txt

#### 1. Prepare the datasets and models

#####CIFAR-10
please download the datasets in the folder "data/cifar10" and models in the "model/cifar10" from https://drive.google.com/drive/folders/1idozSeUa9fHQBdPwMGWmQ7GhZuD3Rtpc?usp=sharing

#####Tiny-ImageNet
please download the datasets in the folder "data/tiny_imagenet" from

#### 2. Generate the query set
```python
CUDA_VISIBLE_DEVICES=0 python configs/tiny_imagenet/conditional.yaml --devices 0 --d=0.1 --alpha=0.1 --beta=0 --gamma=1 --omega=5
```

#### 3. verify the query set
```python
cd evaluation
CUDA_VISIBLE_DEVICES=0 python auc_matching_rate.py
```
