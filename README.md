# 360FusionNeRF: Panoramic Neural Radiance Fields with Joint Guidance
## A method to synthesize novel views from a single 360-degree panorama image based on the neural radiance field (NeRF). This method can render high-resolution 360 camera images based on only one LiDAR Scan and one 360 images. 

<a href="https://www.youtube.com/embed/JN0OsU-92XA" target="_blank"><img src="http://img.youtube.com/vi/JN0OsU-92XA/0.jpg" 
alt="cla" width="640" height="480" border="10" /></a>


<img src="room1.gif" alt="Room1" width="240" height="120"> <img src="room2.gif" alt="Room2" width="240" height="120"> <img src="room3.gif" alt="Room3" width="240" height="120"> <img src="room4.gif" alt="Room4" width="240" height="120">

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
