# OpenTMA: Open Text-Motion Alignment Project

## ✨ Quick Introduction

OpenTMA is a project that aims to provide a simple and efficient way to align text and motion data. It is designed to be easy to use and flexible, allowing users to align text and motion data in the latent space. 


## Todo List


- [x] Release the OpenTMA training.
- [ ] Release the OpenTMA checkpoints.
- [ ] Support PyPI (`pip install opentma`).

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

## Model Training

### 1. Data Preparation

Our OpenTMA project supports three datasets: [HumanML3D](https://github.com/EricGuo5513/HumanML3D?tab=readme-ov-file#how-to-obtain-the-data), [Motion-X](https://motionx.deepdataspace.com/), and [UniMoCap](https://github.com/LinghaoChan/UniMoCap). 

<details>
  <summary><b> HumanML3D Data Preparation </b></summary>

Please following the instructions in the [HumanML3D](https://github.com/EricGuo5513/HumanML3D?tab=readme-ov-file#how-to-obtain-the-data) repository to download and preprocess the data. The data should be stored in the `./datasets/humanml3d` folder. The path tree should look like this:

```
./OpenTMR/datasets/humanml3d/
├── all.txt
├── Mean.npy
├── new_joints/
├── new_joint_vecs/
├── Std.npy
├── test.txt
├── texts/
├── train.txt
├── train_val.txt
└── val.txt
```

</details>


<details>
  <summary><b> Motion-X Data Preparation </b></summary>

Please following the instructions in the [Motion-X](https://github.com/IDEA-Research/Motion-X?tab=readme-ov-file#-dataset-download) project. And then please follow the [HumanTOMATO](https://github.com/IDEA-Research/HumanTOMATO/tree/main/src/tomato_represenation) repository to preprocess the data into `tomatao` format. The data should be stored in the `./datasets/Motion-X` folder. The path tree should look like this:

```
./OpenTMR/datasets/Motion-X
├── mean_std
│   └── vector_623
│       ├── mean.npy
│       └── std.npy
├── motion_data
│   └── vector_623
│       ├── aist/       (subset_*/*.npy)
│       ├── animation/
│       ├── dance/
│       ├── EgoBody/
│       ├── fitness/
│       ├── game_motion/
│       ├── GRAB/
│       ├── HAA500/
│       ├── humanml/
│       ├── humman/
│       ├── idea400/
│       ├── kungfu/
│       ├── music/
│       └── perform/
├── split
│   ├── all.txt
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
└── texts
    ├── semantic_texts
    │   ├── aist/       (subset_*/*.txt)
    │   ├── animation/
    │   ├── dance/
    │   ├── EgoBody/
    │   ├── fitness/
    │   ├── game_motion/
    │   ├── GRAB/
    │   ├── HAA500/
    │   ├── humanml/
    │   ├── humman/
    │   ├── idea400/
    │   ├── kungfu/
    │   ├── music/
    └───└── perform/
```

</details>


<details>
  <summary><b> UniMoCap Data Preparation </b></summary>

Please following the instructions in the [UniMoCap](https://github.com/LinghaoChan/UniMoCap) repository to download and preprocess the data (HumanML3D, BABEL, and KIT-ML). The data should be stored in the `./datasets/UniMocap` folder. The path tree should look like this:

```
./OpenTMR/datasets/UniMocap
├── all.txt
├── Mean.npy
├── new_joints/     (*.npy)
├── new_joint_vecs/ (*.npy)
├── Std.npy
├── test.txt
├── texts/          (*.txt)
├── train.txt
├── train_val.txt
└── val.txt
```

</details>



### 2. Pretrained Checkpoints Used in the Evaluation 

Here, we provide some pre-traind checkpoints for the evaluation. Here are two methods to download the checkpoints:


<details>
<summary><b> Google Drive</b></summary>


Download the checkpoints from the [Google Drive](https://drive.google.com/drive/folders/1aWpJH4KTXsWnxG5MciLHXPXGBS7vWXf7?usp=share_link) and put them in the `./deps` folder. Please unzip the checkpoints via the following command:
```
unzip *.zip
```
Finally, the path tree should look like this:

```
./deps
├── distilbert-base-uncased/
├── glove/
├── t2m/
└── transforms/
```

</details>


<details>
<summary><b> Baidu Drive</b></summary>


Download the checkpoints from the [Baidu Drive](https://pan.baidu.com/s/1SIwGDX2aDWTR4hLhUHrPlw?pwd=evan ) (code: `evan`) and put them in the `./deps` folder. Please unzip the checkpoints via the following command:
```
tar –xvf deps.tar
```
Finally, the path tree should look like this:

```
./deps
├── distilbert-base-uncased/
├── glove/
├── t2m/
└── transforms/
```

</details>


### 3. Downloading Pretrained Checkpoints

We provide some pretrained checkpoints of OpenTMA for evaluation.
