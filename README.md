# Robust Classification via Regression for Learning with Noisy Labels

__DD2412 HT24 Deep Learning, Advanced Course - Shifted Gaussian Noise (SGN), Group 7__

## Overview

This project is a reimplementation and analysis of the method proposed in the paper *"Robust Classification via Regression for Learning with Noisy Labels"* by Englesson and Azizpour (ICLR 2024). The study explores the use of a **Shifted Gaussian Noise (SGN)** model for mitigating the impact of noisy labels in supervised deep learning tasks.

The primary focus is to validate the robustness of SGN in comparison to baseline methods under resource-constrained conditions. The experiments are conducted on the CIFAR-10 dataset with symmetric and asymmetric label noise.

Github: https://github.com/ErikEnglesson/SGN 


## Project structure
```
src/
|
|-- datasets/ 
|   |-- cifar10
|   |-- cifar100  # Not Used!
|
|-- models/
|   |-- wide_resnet_28_2.py
|
|-- methods/
|   |-- ce.py
|   |-- elr.py
|   |-- gce.py
|   |-- het.py  # Not Used!
|   |-- sgn.py
|
|-- experiments/  # Entry points!
|   |-- ce.py
|   |-- elr.py
|   |-- gce.py
|   |-- het.py  # Not Used!
|   |-- sgn.py  # Includes Ablation Study Params
|
|-- utils/
|   |-- config.py
```



## Key Features of SGN

1. **Loss Reweighting (LR):** Adjusts the impact of noisy samples during training by predicting a covariance matrix.
2. **Label Correction (LC):** Dynamically estimates true labels using a Gaussian noise model in a transformed regression space.
3. **Unified Framework:** Combines LR and LC for robust performance under noisy label conditions.



## Methodology
### Dataset
- **CIFAR-10**: 60,000 images, 10 classes.
- Experiments conducted with:
  - Symmetric noise levels: 0%, 20%, 40%.
  - Asymmetric noise levels: 20%, 40%.

### Models Implemented
- **Cross-Entropy (CE):** Standard classification baseline.
- **Generalized Cross-Entropy (GCE):** Incorporates robustness parameter for noisy labels.
- **Early Learning Regularization (ELR):** Prevents overfitting noisy labels.
- **Shifted Gaussian Noise (SGN):** Core method under evaluation.

### Extensions
- Temporal analysis over training epochs to assess learning dynamics.
- Complex asymmetric noise label mappings to reflect real-world conditions.



## Results
### Performance Highlights
- **SGN** demonstrated robust performance under both symmetric and asymmetric noise.
- Outperformed CE and ELR consistently, particularly in high asymmetric noise scenarios.
- **GCE** showed slightly better performance in symmetric noise but overfit in asymmetric conditions.

### Ablation Study
- Loss Reweighting and Label Correction were identified as critical components of SGN's robustness.
- Disabling either component resulted in reduced stability and accuracy.

### Key Observations
- SGN uniquely avoids overfitting in late training stages.
- Early stopping can improve results for some methods but introduces complexity and instability.



## Conclusion
This reimplementation validates SGN as a promising solution for learning with noisy labels. Despite resource constraints and reduced training epochs, SGN exhibited stable and robust performance - proving to be less sensitive to noise and holding a stable learning curve. 

### Future Directions
- Extending evaluations to larger datasets like CIFAR-100 or ImageNet.
- Exploring dynamic noise modeling and semi-supervised learning integration.
- Adopting standardized robustness metrics for comprehensive evaluation.



## References
- Englesson, E., & Azizpour, H. (2024). *Robust Classification via Regression for Learning with Noisy Labels*. [Paper Link](https://openreview.net/pdf?id=wfgZc3IMqo)
- CIFAR-10 Dataset: [Website](https://www.cs.toronto.edu/~kriz/cifar.html)



## Authors
- Oskar Wallberg (oskarew@kth.se)
- Johannes Rosing (jrosing@kth.se)