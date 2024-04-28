# OpenTMA: Open Text-Motion Alignment Project

## ✨ Quick Introduction

OpenTMA is a project that aims to provide a simple and efficient way to align text and motion data. It is designed to be easy to use and flexible, allowing users to align text and motion data in the latent space. 

## 🚀 Quick start

### Installation

```bash
pip install opentma
```

### Usage

```python
# Load text and motion data
import torch
from transformers import AutoTokenizer, AutoModel
from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from collections import OrderedDict

modelpath = 'distilbert-base-uncased'

textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4)
motionencoder = ActorAgnosticEncoder(nfeats=126, vae = True, num_layers=4)

"""
load model here
"""

motion = torch.randn(1, 64, 126)    # B = 1, T = , D = , need normalization
lengths = [64]
print(textencoder(["a man is running"]).loc)
print(motionencoder(motion, lengths).loc)
```

## Test for Evaluation

Before running the code below, please revise the `retreival.sh` (like `path1` variable) file to set the correct path for the data. 

```bash
bash retreival.sh
```

For convenience, the output format is in the form of markdown. 