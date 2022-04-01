# PoreNet

PyTorch demo code of [High-resolution Face Recognition via Deep Pore-feature Matching](https://ieeexplore.ieee.org/abstract/document/8803686).

### Example result
![result](https://github.com/johnnysclai/PoreNet/img/billgates_result.png)

### Citation
If you find this work useful for your research, please consider cite our paper:
```
@inproceedings{lai2019high,
  title={High-resolution Face Recognition via Deep Pore-feature Matching},
  author={Lai, Shun-Cheung and Kong, Minna and Lam, Kin-Man and Li, Dong},
  journal={IEEE International Conference on Image Processing},
  pages={1173-1177},
  year={2019},
  month={Sep.}
}
```

### Requirements
- Python 3
- Jupter notebook
- numpy
- matplotlib
- PyTorch
- OpenCV 3
- scikit-image
- dill

Tested on MacOS with Python 3.9.7(anaconda3) and torch==1.11.0, dill==0.3.4, numpy==1.20.3, opencv-python==4.5.5.64, scikit-image==0.18.3.

## Reference
- GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence: [code](https://github.com/JiawangBian/GMS-Feature-Matcher) and [paper](https://ieeexplore.ieee.org/document/8099785)
