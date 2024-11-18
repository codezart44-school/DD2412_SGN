# DD2412 HT24 Deep Learning, Advanced Course - Shifted Gaussian Noise (SGN), Group 7
Reimplementation and Extension of paper ROBUST CLASSIFICATION VIA REGRESSION FOR LEARNING WITH NOISY LABELS (2024) (https://openreview.net/pdf?id=wfgZc3IMqo) by Erik Englesson &amp; Hossein Azizpour. Final Project in DD2412 HT24 Deep Learning, Advanced Course (DLAHT22)

Github: https://github.com/ErikEnglesson/SGN 

### Datasets
* Cifar10
* Cifar100
* ?Cloathing1M (Large and cumbersome)

## Methods for handling noise
### Baseline Methods
* Cross-Entropy (CE) as a fundamental baseline.
* Generalized Cross-Entropy (GCE) for its robustness to noisy labels.
* Label Smoothing (LS), which aligns with our method's use for transforming to a compositional dataset.

These baselines align well with our project scope and ensure a meaningful comparison without overextending our resources. 

### Other Methods
* Heteroscedastic Noise (HET) and Noise Against Noise (NAN) approaches that use Gaussian noise models similar to our loss reweighting.
* Early-Learning Regularization (ELR), Symmetric and Asymmetric Optimization (SOP), and Noise Attention Learning (NAL) which are relevant due to their loss reweighting or label correction features.

## Extension Plan:
We have decided to focus on extension option [1] self-supervised pretraining as it actually seems relevant to the task of classification via regression with noisy labels, and we want to investigate how self-supervised pretraining could synergize with the robustness-methods against noisy labels mentioned above. This involves pretraining with contrastive learning methods to acquire robustness when fine-tuning with noisy labels. We ask ourselves: Will pretrainign be effective in this context of noisy labels? Do we see better or faster convergence or both? 

## Experiments: 
* Baselines comparison: Quantitative comparison of this implementation and other methods for label noise robustness. For fair assessment we want to include datasets many other models have been tested on. 
* Noise level variation: Test the implementations under many levels of noise (10% … 90%) to see method performance changes as noise levels increase / decrease. 
* Noise type variation: Assess how these methods’ performances are impacted by symmetric contra asymmetric noise in the labels (noise scattered evenly among all classes or concentrated to one or a few classes at a time). 
* Model scalability: Implement these methods for models of different scales (ResNet-18 … ResNet-50) to assess how well these methods extend to more complex and sophisticated network architectures. Assess how computational efficiency is impacted - does the increase in performance justify the computational cost?  

