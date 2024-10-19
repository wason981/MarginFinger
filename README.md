# MarginFinger--Official Pytorch Implementation

![Framework](https://github.com/wason981/MarginFinger/blob/main/images/framework.png)
**MarginFinger: Controlling Generated Fingerprint Distance to Classification boundary Using Conditional GANs**<br>
Weixing Liu,Shenghua Zhong<br>
Paper:https://dl.acm.org/doi/10.1145/3652583.3658058<br>

Abstract:*Deep neural networks (DNNs) are widely employed across various domains, with their training costs making them crucial assets for model owners. However, the rise of Machine Learning as a Service has made models more accessible, but also increases the risk of leakage. Attackers can successfully steal models through internal leaks or API access, emphasizing the critical importance of protecting intellectual property. Several watermarking methods have been proposed, embedding secret watermarks of model owners into models. However, watermarking requires tampering with the model's training process to embed the watermark, which may lead to a decrease in utility. Recently, some fingerprinting techniques have emerged to generate fingerprint samples near the classification boundary to detect pirated models. Nevertheless, these methods lack distance constraints and suffer from high training costs. To address these issues, we propose to utilize conditional generative network to generate fingerprint data points, enabling a better exploration of the model's decision boundary. By incorporating margin loss during GAN training, we can control the distance between generated data points and classification boundary to ensure the robustness and uniqueness of our method. Moreover, our method does not require additional training of proxy models, enhancing the efficiency of fingerprint acquisition. To validate the effectiveness of our approach, we evaluate it on CIFAR-10 and Tiny-ImageNet, considering three types of model extraction attacks, fine-tuning, pruning, and transfer learning attacks. The results demonstrate that our method achieves ARUC values of 0.186 and 0.153 on CIFAR-10 and Tiny-ImageNet datasets, respectively, representing a remarkable improvement of 400% and 380% compared to the current leading baseline.*

## Requirements
* Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.8 installation. We recommend Anaconda3 with numpy 1.21 or newer.
* We recommend Pytorch 2.0.1, which we used for all experiments in the paper.

## 1. Prepare the datasets
You can download our pre-splited data set in `data` for fingerprint generation and verification. The defender data set belongs to the model defender and is used for source model training and fingerprint GAN model training, while the attacker training set is used to simulate the attacker training to steal the model.

**CIFAR-10**. please download the datasets in the folder "data/cifar10" and models in the "model/cifar10" from [here](https://drive.google.com/drive/folders/1idozSeUa9fHQBdPwMGWmQ7GhZuD3Rtpc?usp=sharing)

**Tiny-ImageNet**. please download the datasets in the folder "data/tiny_imagenet" from [here](https://drive.google.com/drive/folders/1AvUa1A3bxqRHDizjH9dw4XDTGucpf3F4?usp=sharing)

## Generation Stage 
### 1. Training networks
```.bash
# the trained model will put into the folder `model`
python train.py configs/tiny_imagenet/conditional.yaml --devices 0 --d=0.1 --alpha=0.1 --beta=0 --gamma=1 --omega=5
```
### 2. Fingerprint Generaion
```.bash 
#You can use the trained model to generated fingerprints.
python sample.py
```

## 3. verification Stage
```.bash
#You should first go to the evaluation and then verify.
cd evaluation
python auc_matching_rate.py
```
## Citing this work
If you use this repository for academic research, you are highly encouraged (though not required) to cite our paper:
```
@inproceedings{10.1145/3652583.3658058,
author = {Liu, Weixing and Zhong, Shenghua},
title = {MarginFinger: Controlling Generated Fingerprint Distance to Classification boundary Using Conditional GANs},
year = {2024},
url = {https://doi.org/10.1145/3652583.3658058},
doi = {10.1145/3652583.3658058},
}
```
