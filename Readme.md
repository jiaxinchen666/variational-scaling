# Variational Metric Scaling for Metric-Based Meta-Learning

### Dataset

Download the images: https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE

Make a folder materials/images and put those images into it.

### Requirements

scripy==1.1.0

pytorch==0.4.0

### Train

SVS

`python train.py`

D-SVS

`python train_multiscale.py`

D-AVS

`python train_gen.py`

### Evaluation

SVS

`python test.py --multi False`

D-SVS

`python test.py --multi True`

D-AVS

`python test_gen.py`

### Note

This code is built on https://github.com/cyvius96/prototypical-network-pytorch
