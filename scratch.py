from mmaction.datasets import build_dataset, build_dataloader
from mmcv import Config
cfg = Config.fromfile('configs/shotout_i3d_32x1.py')
ds = build_dataset(cfg.data.train)
ds_loader = build_dataset(ds)