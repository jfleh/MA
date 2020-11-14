# -*- coding: utf-8 -*-
import os
from math import ceil
import numpy as np
from odl.discr.lp_discr import uniform_discr
from odl.discr.grid import RectGrid
from odl.discr.partition import uniform_partition_fromgrid
from odl.tomo import Parallel2dGeometry, RayTransform
from skimage.transform import resize
from dival.datasets.lodopab_dataset import LoDoPaBDataset
from dival.datasets.dataset import Dataset as DivalDataset


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class LoDoPaBSmallDataset(DivalDataset):
    def __init__(self, size=128, angle_step=1, detector_pixel_step=1,
                 impl='astra_cuda', **kwargs):
        self.lodopab = LoDoPaBDataset(impl=impl, **kwargs)
        self.num_elements_per_sample = 2
        domain = uniform_discr(self.lodopab.space[1].min_pt,
                               self.lodopab.space[1].max_pt, (size, size),
                               dtype=np.float32)
        self.angle_step = angle_step
        self.detector_pixel_step = detector_pixel_step
        self.train_len = self.lodopab.train_len
        self.validation_len = self.lodopab.validation_len
        self.test_len = self.lodopab.test_len
        self.random_access = True

        num_angles = ceil(self.lodopab.shape[0][0] /
                          self.angle_step)
        num_detector_pixels = ceil(self.lodopab.shape[0][1] /
                                   self.detector_pixel_step)
        apart = uniform_partition_fromgrid(
            RectGrid(self.lodopab.geometry.angles[::self.angle_step]))
        dpart = uniform_partition_fromgrid(
            self.lodopab.geometry.det_grid[::self.detector_pixel_step])
        self.geometry = Parallel2dGeometry(apart=apart, dpart=dpart)
        self.shape = ((num_angles, num_detector_pixels), (size, size))
        range_ = uniform_discr(self.geometry.partition.min_pt,
                               self.geometry.partition.max_pt,
                               self.shape[0], dtype=np.float32)
        super().__init__(space=(range_, domain))
        self.ray_trafo = self.get_ray_trafo(impl=impl)

    def get_ray_trafo(self, impl=None):
        return RayTransform(self.space[1], self.geometry, range=self.space[0],
                            impl=impl)

    def get_sample(self, index, part='train', out=None):
        if out is None:
            out = (True, True)
        out_ = (not (isinstance(out[0], bool) and not out[0]),
                not (isinstance(out[1], bool) and not out[1]))
        lodopab_sample = self.lodopab.get_sample(index, part=part, out=out_)
        (obs, gt) = (None, None)
        if lodopab_sample[0] is not None:
            obs = self.space[0].element(np.asarray(lodopab_sample[0])[
                    ::self.angle_step, ::self.detector_pixel_step])
        if lodopab_sample[1] is not None:
            gt = self.space[1].element(
                resize(np.asarray(lodopab_sample[1]), self.shape[1], order=1))
        return (obs, gt)

    def get_samples(self, key, part='train', out=None):
        if out is None:
            out = (True, True)
        out_ = (not (isinstance(out[0], bool) and not out[0]),
                not (isinstance(out[1], bool) and not out[1]))
        lodopab_samples = self.lodopab.get_samples(key, part=part, out=out_)
        (obs, gt) = (None, None)
        if lodopab_samples[0] is not None:
            obs = np.asarray(lodopab_samples[0])[
                :, ::self.angle_step, ::self.detector_pixel_step]
        if lodopab_samples[1] is not None:
            gt = np.empty((lodopab_samples[1].shape[0],) + self.shape[1])
            for i in range(gt.shape[0]):
                gt[i] = resize(np.asarray(lodopab_samples[1][i]),
                               self.shape[1], order=1)
        return (obs, gt)


if __name__ == '__main__':
    SIZE = 128
    ANGLE_STEP = 20
    DETECTOR_PIXEL_STEP = 2

    lodopab_dival_dataset = LoDoPaBSmallDataset(
        size=SIZE,
        angle_step=ANGLE_STEP,
        detector_pixel_step=DETECTOR_PIXEL_STEP)

    num_angles = lodopab_dival_dataset.shape[0][0]
    ray_trafo = lodopab_dival_dataset.get_ray_trafo(impl='astra_cuda')  # other
#    options are 'astra_cpu' and 'skimage'

    lodopab_dataset = lodopab_dival_dataset.create_torch_dataset(
        part='train', reshape=((1, num_angles, -1), (1, SIZE, SIZE)))

    (y, x) = lodopab_dataset[0]  # get a sample from dataset (by index)
    print('x.shape: {}'.format(x.shape))
    print('y.shape: {}'.format(y.shape))
