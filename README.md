# NeurAR: Neural Uncertainty for Autonomous 3D Reconstruction with Implicit Neural Representations

Yunlong Ran, Jing Zeng, Shibo He, Lincheng Li, Yingfeng Chen, Gimhee Lee, Jiming Chen, Qi Ye

arXiv:[[2207.10985\] NeurAR: Neural Uncertainty for Autonomous 3D Reconstruction with Implicit Neural Representations (arxiv.org)](https://arxiv.org/abs/2207.10985)

This is the official repository for our paper, NeurAR, we release uncertainty verification part and the planned dataset. The planner and the unity module will not release at the current time.

# Environment setup

To start, we prefer creating the environment using conda:

```
conda env create -f environment.ymal

conda activate neurar
```

Alternatively, you can install them yourself

imageio

imageio-ffmpeg

[Pytorch](https://pytorch.org/)

colorlog

matplotlib

configargparse

tqdm

opencv-python

pandas

jupyter

seaborn

numpy

scikit-image

lpips

# Getting the data

- for the NeRF synthetic data

```
cd data
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip nerf_example_data.zip
```

* for our collected data, you can download from [here](https://drive.google.com/file/d/1Hp5cz1dr7h6ucRNTCvbpisnsdC5XItBm/view?usp=sharing). 

# Quick Start

To verify uncertainty:

```
cd src
python verify_uncertainty.py --config ../configs/lego.txt
```

Then launch your jupyter note book and follow the link in your browser

```
jupyter notebook
```

Run ```verify uncertainty.ipynb``` jupyter scripts one by one.

For other scene, you can simple replace ```lego.txt``` with {scene}.txt

# Evaluation on planned model

You can download planned and trained model [here](https://drive.google.com/file/d/1AJbQlcifoBNrZgbiFuoVcAakigPTqoUv/view?usp=sharing) and unzip them into ```/logs```.

And then run:

```
cd src
python eval.py --config ../configs/cabin.txt
```

Then run ```eval.ipynb``` jupyter scripts one by one to get metrics.

# Citation

```
@ARTICLE{10012495,
  author={Ran, Yunlong and Zeng, Jing and He, Shibo and Chen, Jiming and Li, Lincheng and Chen, Yingfeng and Lee, Gimhee and Ye, Qi},
  journal={IEEE Robotics and Automation Letters}, 
  title={NeurAR: Neural Uncertainty for Autonomous 3D Reconstruction With Implicit Neural Representations}, 
  year={2023},
  volume={8},
  number={2},
  pages={1125-1132},
  doi={10.1109/LRA.2023.3235686}}
```