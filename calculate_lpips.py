import numpy as np
from PIL import Image
from run_nerf_helpers import lpips
import torch



rgb_ref = np.asarray(Image.open('/home/skulkarni/vit_loss/omni_nerf/data/rep3/rep3_rgb.png').convert('RGB')) / 255.0

rgb_pred = np.asarray(Image.open('/home/skulkarni/vit_loss/omni_nerf/logs/rep3_vit/testset_120000/010.png').convert('RGB')) / 255.0


rgb_pred = rgb_pred*2 - 1
rgb_ref = rgb_ref*2 - 1


rgb_pred = torch.unsqueeze(torch.tensor(rgb_pred).permute(2, 0, 1), 0)
rgb_ref = torch.unsqueeze(torch.tensor(rgb_ref).permute(2, 0, 1), 0)

lpips_metric = lpips(rgb_ref.float(), rgb_pred.float())

print(lpips_metric)