# FCHD-Fully-Convolutional-Head-Detector
Code for FCHD - A fast and accurate head detector

This is the code for FCHD - A Fast and accurate head detector. The manuscript is under review in a journal. The full code is implemented in Python with PyTorch framework. 

## Dependencies

- install PyTorch >=0.4 with GPU (code are GPU-only), refer to [official website](http://pytorch.org)

- install cupy, you can install via `pip install cupy-cuda80` or(cupy-cuda90,cupy-cuda91, etc).

- install visdom for visualization, refer to their [github page](https://github.com/facebookresearch/visdom)

## Installation
1) Install Pytorch

2) Clone this repository
  ```Shell
  git clone https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector
  ```
3) Build cython code for speed:
  ```Bash
  cd src/nms/
  python build.py build_ext --inplace
  ```
4) Start visdom server for visualization:
```Bash
python -m visdom.server
```


