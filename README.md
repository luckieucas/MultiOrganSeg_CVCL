# MultiOrganSeg_CVCL
code for the paper: Context-aware Voxel-wise Contrastive Learning for Label Efficient Multi-organ Segmentation
## Setup

### Requirements

```bash
python 3.7.8
pytorch 1.12.1
CUDA 10.2
```


## Running

### Training

```bash
python train_model.py --config_file train_config.yaml
```

### Testing

```bash
python test_model.py --config_file test_config.yaml
```


## Citing this work
```

```

# Acknowledgement
Thanks [Partially-supervised-multi-organ-segmentation](https://github.com/MIRACLE-Center/Partially-supervised-multi-organ-segmentation) and [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for their wonderfurl work. Part of the code is borrowed from them. Please feel free to cite their work:
```
@article{shi2021marginal,
  title={Marginal loss and exclusion loss for partially supervised multi-organ segmentation},
  author={Shi, Gonglei and Xiao, Li and Chen, Yang and Zhou, S Kevin},
  journal={Medical Image Analysis},
  volume={70},
  pages={101979},
  year={2021},
  publisher={Elsevier}
}

@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group US New York}
}


```

