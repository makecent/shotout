import copy
import os

import numpy as np
import torch
from mmaction.datasets import DATASETS, PIPELINES, BaseDataset
from mmaction.datasets.pipelines import Compose
from mmaction.models import LOSSES
from mmaction.models.losses import BaseWeightedLoss
from torch.nn import functional as F
from torch.utils.data import Dataset


@DATASETS.register_module()
class Shotout(BaseDataset):
    def __init__(self, raw_video=False, filename_tmpl='img_{:05}.jpg', *args, **kwargs):
        self.raw_video = raw_video
        self.filename_tmpl = filename_tmpl
        super(Shotout, self).__init__(*args, **kwargs)

    def load_annotations(self):
        video_infos = []
        class_label = {'challenge': 0, 'play': 1, 'throwin': 2}
        with open(self.ann_file, 'r') as f:
            video_frames = {}
            for line in f.readlines():
                line = line.strip().split(',')
                if line[2] in ['challenge', 'play', 'throwin']:
                    sample_info = dict(time=float(line[1]),
                                       label=class_label[line[2]])
                    if self.raw_video:
                        sample_info.update(dict(filename=os.path.join(self.data_prefix, line[0] + '.mp4')))
                    else:
                        frame_dir = os.path.join(self.data_prefix, line[0])
                        total_frames = video_frames.setdefault(line[0], len(os.listdir(frame_dir)))
                        sample_info.update(dict(frame_dir=frame_dir, total_frames=total_frames))
                    video_infos.append(sample_info)
        return video_infos

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = 'RGB'
        results['start_index'] = self.start_index
        return self.pipeline(results)


@PIPELINES.register_module()
class ShotoutSampleFrames:
    def __init__(self,
                 clip_len=16,
                 frame_interval=1,
                 sampling_style='center',
                 jitter_magnitude=0,
                 test_mode=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.sampling_style = sampling_style
        self.jitter = jitter_magnitude
        self.test_mode = test_mode

    def __call__(self, results):
        """Perform the FetchFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        frame_index = int(results['time'] * 25)
        if self.sampling_style == 'center':
            frame_inds = frame_index + np.arange(-self.clip_len / 2, self.clip_len / 2, dtype=int) * self.frame_interval
        elif self.sampling_style == 'right':
            frame_inds = frame_index + np.arange(0, self.clip_len, dtype=int) * self.frame_interval
        elif self.sampling_style == 'left':
            frame_inds = frame_index + np.arange(-self.clip_len + 1, 1, dtype=int) * self.frame_interval
        else:
            raise ValueError('sampling style not recognized')
        if self.jitter > 0:
            frame_inds += np.random.randint(-self.jitter, self.jitter)
        frame_inds = np.clip(frame_inds, 1, results['total_frames'])

        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1

        return results


@LOSSES.register_module()
class FocalLoss(BaseWeightedLoss):
    def __init__(self, gamma=2.0, alpha=0.25, do_onehot=True, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.do_onehot = do_onehot
        self.label_smoothing = label_smoothing

    @staticmethod
    def _smooth(label, smoothing=0.1):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            label = label * (1.0 - smoothing) + (label * smoothing).mean(dim=-1, keepdim=True)
        return label

    def _forward(self, cls_score, label, **kwargs):
        if self.do_onehot:
            label = F.one_hot(label, num_classes=cls_score.size(-1))
        loss = F.binary_cross_entropy_with_logits(cls_score, self._smooth(label, self.label_smoothing), **kwargs)
        cls_score = cls_score.sigmoid()
        pt = (1 - cls_score) * label + cls_score * (1 - label)
        focal_weight = (self.alpha * label + (1 - self.alpha) * (1 - label)) * pt.pow(self.gamma)
        return focal_weight * loss
