# From Sequences to Images: Deep Feature Extraction for Transposable Element Classification

This repository contains the implementation of the framework described in our paper:

> **From Sequences to Images: Deep Feature Extraction for Transposable Element Classification**
> *(Manuscript under review / in press)*

---

## 🧬 Overview

We introduce a novel approach for **transposable element (TE) classification** by converting genomic sequences into **2D image representations**. This strategy enables the use of state-of-the-art **convolutional neural networks (CNNs)**—such as **ResNet50RS** and **InceptionV3**—combined with **classical machine learning classifiers**.

### Key Contributions

* **2D Sequence-to-Image Mapping**
  A scalable and deterministic method to transform DNA sequences into image-like representations.

* **Deep Feature Extraction**
  Use of both general-purpose CNNs and domain-specific architectures (e.g., **TERL**) to extract discriminative features.

* **Hybrid Classification Models**
  Integration of deep features with classical machine learning algorithms (e.g., **Random Forest**, **SVM**) to achieve superior performance in complex taxonomic scenarios.

---

## 📊 Datasets & Databases

Experiments were conducted on **five taxonomic datasets (DS1–DS5)**. All datasets include an additional **non-TE class** composed of shuffled sequences, ensuring robustness against noise and false positives.

### Dataset Summary

| Dataset | Scope       | Target Classes                                                                        | Source      |
| ------: | ----------- | ------------------------------------------------------------------------------------- | ----------- |
| **DS1** | Superfamily | Copia, Gypsy, Bel-Pao, ERV, L1, Tc1-Mariner, hAT                                      | RepBase     |
| **DS2** | Order       | Class II, LTR, LINE                                                                   | RepBase     |
| **DS3** | Superfamily | Copia, Gypsy, Bel-Pao, ERV, L1, SINE, Tc1-Mariner, hAT, Mutator, PIF-Harbinger, CACTA | 7 Databases |
| **DS4** | Order       | Class II, LTR, LINE, SINE                                                             | 7 Databases |
| **DS5** | Order       | Class II, LTR, LINE                                                                   | Mixed*      |

* **DS5** uses RepBase for training and independent databases for testing, enabling evaluation of **cross-dataset generalization**.

---

## 🛠️ Quick Start

### 1. Data Organization

The image generator expects the following directory structure:

```text
Data/
└── Dataset1/
    ├── Train/
    │   └── LTR.fa
    └── Test/
        └── LTR.fa
```

Each class must be provided as a separate **FASTA (.fa)** file.

---

### 2. Generating Images from Sequences

To convert genomic sequences into 2D images, use the script `block-6.py`. The script processes FASTA files and produces the corresponding image-based representations.

**Execution example:**

```bash
python block-6.py \
  -din Data/Dataset1 \
  -dout generated-images/Dataset1 \
  -size 456
```

**Parameters:**

* `-din`: Input directory containing FASTA files
* `-dout`: Output directory for generated images
* `-size`: Image resolution (e.g., 456 × 456)

---

## 🗄️ Databases Used

This work integrates TE sequences from the following specialized databases:

* **RepBase** — [https://doi.org/10.1159/000084979](https://doi.org/10.1159/000084979)
* **DPTEdb** — [https://doi.org/10.1093/database/baw078](https://doi.org/10.1093/database/baw078)
* **SPTEdb** — [https://doi.org/10.1093/database/bay024](https://doi.org/10.1093/database/bay024)
* **PGSB PlantsDB** — [https://doi.org/10.1093/nar/gkv1130](https://doi.org/10.1093/nar/gkv1130)
* **RiTE Database** — [https://doi.org/10.1186/s12864-015-1762-3](https://doi.org/10.1186/s12864-015-1762-3)
* **TREP** — [https://doi.org/10.1016/S1360-1385(02)02372-5](https://doi.org/10.1016/S1360-1385%2802%2902372-5)
* **TEfam** — [https://doi.org/10.1186/1471-2164-12-260](https://doi.org/10.1186/1471-2164-12-260)

---

## 📚 Citation

If you use this code or methodology, please cite:

```bibtex
@article{VieiraW2026,
  title   = {From Sequences to Images: Deep Feature Extraction for Transposable Element Classification},
  author  = author = {Viera, Wellington de Souza and Góes, Fabiana Rodrigues and Saito, Priscila Tiemi Maeda and Bugatti, Pedro Henrique},
  journal = {Under Review},
  year    = {2026}
}
```
