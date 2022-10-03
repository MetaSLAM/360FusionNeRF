# 360FusionNeRF: Panoramic Neural Radiance Fields with Joint Guidance


<a href="https://www.youtube.com/embed/JN0OsU-92XA" target="_blank"><img src="http://img.youtube.com/vi/JN0OsU-92XA/0.jpg" 
alt="cla" width="640" height="480" border="10" /></a>

## Overview

[[arXiv]](https://arxiv.org/abs/2209.14265)

 We present a method to synthesize novel views from a single 360∘ panorama image based on the neural radiance field (NeRF). Prior studies in a similar setting rely on the neighborhood interpolation capability of multi-layer perceptions to complete missing regions caused by occlusion, which leads to artifacts in their predictions. We propose 360FusionNeRF, a semi-supervised learning framework where we introduce geometric supervision and semantic consistency to guide the progressive training process. Firstly, the input image is re-projected to 360∘ images, and auxiliary depth maps are extracted at other camera positions. The depth supervision, in addition to the NeRF color guidance, improves the geometry of the synthesized views. Additionally, we introduce a semantic consistency loss that encourages realistic renderings of novel views. We extract these semantic features using a pre-trained visual encoder such as CLIP, a Vision Transformer trained on hundreds of millions of diverse 2D photographs mined from the web with natural language supervision. Experiments indicate that our proposed method can produce plausible completions of unobserved regions while preserving the features of the scene. When trained across various scenes, 360FusionNeRF consistently achieves the state-of-the-art performance when transferring to synthetic Structured3D dataset (PSNR\~5%, SSIM\~3% LPIPS\~13%), real-world Matterport3D dataset (PSNR\~3%, SSIM\~3% LPIPS\~9%) and Replica360 dataset (PSNR\~8%, SSIM\~2% LPIPS\~18%).



## Setup

We use the following folder structure:
```
logs/ (images, videos, checkpoints)
data/
  std/
configs/ (run configuration files)
CLIP/ (Fork of OpenAI's clip repository with a wrapper)
```

Create conda environment:
```
conda create -n 360fusionnerf python=3.9
conda activate 360fusionnerf
```

Set up requirements and our fork of CLIP:
```
pip install -r requirements.txt
cd CLIP
pip install -e .
```

## How to run
```
python run_nerf.py --config configs/st3d.txt
```

You could also generate data from your own panorama.
See the file [generate.py](https://github.com/MetaSLAM/360FusionNeRF/tree/main/generate_data/generate.py) for more details.

## Citation
If you find our work useful in your research, please consider citing:

	@article{shreyas2022nerf360,
	  title={360FusionNeRF: Panoramic Neural Radiance Fields with Joint Guidance},
	  author={Shreyas Kulkarni, Peng Yin, and Sebastian Scherer},
	  journal={arXiv preprint arXiv:2209.14265},
	  year={2022}
	}
   
The code is based on [DietNeRF](https://github.com/ajayjain/DietNeRF) and [OmniNeRF](https://github.com/cyhsu14/OmniNeRF) codes.
