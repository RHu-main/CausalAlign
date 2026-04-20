# Anonymous Repository for Paper Submission

This anonymous repository contains the official implementation of a causal framework for **Multi-Modal Zero-Shot Skeleton Action Recognition (MM-ZSSAR)**.

## Overview

Multi-modal zero-shot skeleton action recognition benefits from complementary textual semantics and visual motion cues, but its performance is often limited by **alignment bias** caused by confounders. In particular, two types of confounders are considered in this project:

- **Observable confounders**, mainly arising from modality heterogeneity between skeleton and text representations.
- **Latent confounders**, mainly arising from spurious correlations induced by highly similar skeleton patterns.

To address this issue, we propose a causal framework that introduces explicit causal intervention into cross-modal representation learning. The framework is composed of three collaborative modules:

- **SCR**: Spurious Correlations Removal
- **HFC**: Heterogeneous Feature Calibrator
- **CFA**: Causal Feature Alignment

Together, these modules aim to mitigate alignment bias, harmonize heterogeneous modalities, and improve generalization to unseen action classes.

---

## Features

- Causal formulation for mitigating alignment bias in MM-ZSSAR
- Joint modeling of observable and latent confounders
- Modular design with SCR, HFC, and CFA
- Support for experiments on standard benchmark datasets
- Reproducible training and evaluation pipeline

---

## Code Structure
* We set basic configuration in config.py and use <python main_mm.py> to start modal training and evaluation.
* In <tool.py> we compute causal logits by method **create_logits_causal**, and the SCR\HFC\CFA are defined in utils.py.
* Note that the data split we have followed the protocol of [SMIE](https://github.com/YujieOuO/SMIE).

---

## Aknowledge
We thanks to the community that provide full strucutre of this code repo:

```bibtex
@article{zhou2025pgfa,
  title={Zero-Shot Skeleton-Based Action Recognition With Prototype-Guided Feature Alignment},
  author={Kai Zhou and Shuhai Zhang and Zeng You and Jinwu Hu and Mingkui Tan and Fei Liu},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  volume={34},
  pages={4602-4617},
  publisher={IEEE},
  doi={10.1109/TIP.2025.3586487}
}
```




