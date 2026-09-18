# VS-Splat: Voxel-Selective feed-forward Gaussian Splatting for end-to-end 3D object reconstruction from sparse-views

Official implementation of VS-Splat. Check out the paper on [arXiv](https://arxiv.org/abs/2609.12343v1) and more qualitative results on the [project page](https://vs-splat.github.io/).




## Installation

Our environment is based on **Python 3.10**, **PyTorch 2.3.1**, and **CUDA 11.8**.
Clone the repository and set up the conda environment:

```bash
git clone https://github.com/skkuhyuk/vssplat.git
cd vssplat

conda env create -f vssplat_env.yaml
conda activate vssplat
```

## Datasets


| Dataset | Source |
| --- | --- |
| GObjaverse | [LaRa](https://github.com/autonomousvision/LaRa) |
| GSO | [LaRa](https://github.com/autonomousvision/LaRa) |
| CO3D| [GenerativeDensification](https://github.com/stnamjef/GenerativeDensification) |

A download helper is included:

```bash
python tools/download_dataset.py
```


## Training

Edit training settings at `configs/vs-splat.yaml`. We recommend to leverage batch size=16 with 4 80GB A100 GPUs as written in paper for reproduction.

```bash
python train_lightning.py
```




## Evaluation

Set `infer.ckpt_path` and the dataset path in `configs/infer.yaml`. To reproduce the full benchmark, fill in the checkpoint and dataset paths at the top of `eval_all.py` and run:

```bash
python eval_all.py
```

## Weights 



## Citation



```bibtex
@article{vs-splat,
  title     = {VS-Splat: Voxel-Selective feed-forward Gaussian Splatting for end-to-end 3D object reconstruction from sparse-views},
  author    = {Yunsu, Jeong and Hyuk, Heo and Youngsang, Kwak and Jaehwa, Kwak and Il Yong, Chun},
  journal = {arXiv preprint arXiv:2609.12343},
  year      = {2026}
}
```


