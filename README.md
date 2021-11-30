# MVAE-Pytorch For Marketing Campaign
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Paper
- [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/pdf/1802.05814.pdf)
- [On Mitigating Popularity Bias in Recommendations via
Variational Autoencoders](https://homepages.tuni.fi/konstantinos.stefanidis/docs/sac2021.pdf)

## Dataset
- A Purchase Dataset in S3

## How To Run
### 1. Install
```bash
pip3 install -r requirements.txt
```

### 2. Load Data
```bash
python3 load_data.py
```

### 3. Preprocess
```bash
python3 preprocess.py
```

### 4. Train
```bash
python3 train.py
```

### 5. Inference
```bash
python3 inference.py
```
