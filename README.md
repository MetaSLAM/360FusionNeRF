# 360FusionNeRF: Panoramic Neural Radiance Fields with Joint Guidance

[Paper]()

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

The code is based on [DietNeRF](https://github.com/ajayjain/DietNeRF) and [OmniNeRF](https://github.com/cyhsu14/OmniNeRF) codes.