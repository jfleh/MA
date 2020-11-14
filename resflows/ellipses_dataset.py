import os
from itertools import repeat
import numpy as np
from odl.discr.lp_discr import uniform_discr
from odl.tomo import parallel_beam_geometry, RayTransform
from odl.phantom import ellipsoid_phantom
import torch
from torch.utils.data import Dataset


os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def pair_generator(ray_trafo, seed=None, noise_stddev=0.01, noise_seed=None):
    r = np.random.RandomState(noise_seed)
    sino = ray_trafo.range.element()
    for im in ellipses_generator(size=ray_trafo.domain.shape[0], seed=seed):
        ray_trafo(im, out=sino)
        noisy_sino = r.normal(loc=sino, scale=noise_stddev)
        yield (ray_trafo.range.element(noisy_sino), im)


def ellipses_generator(size=32, seed=None):
    """Yield random ellipse phantom images using
    `odl.phantom.ellipsoid_phantom`.

    Parameters
    ----------
    seed : int or `None`
        initial seed used for random values

    Yields
    ------
    image : odl element
        Random ellipse phantom image with values in [0., 1.].
    """
    space = uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                          shape=(size, size), dtype=np.float32)
    r = np.random.RandomState(seed)
    n_ellipse = 50
    ellipsoids = np.empty((n_ellipse, 6))
    for _ in repeat(None):
        v = (r.uniform(-.5, .5, (n_ellipse,)) *
             r.exponential(.4, (n_ellipse,)))
        a1 = .2 * r.exponential(1., (n_ellipse,))
        a2 = .2 * r.exponential(1., (n_ellipse,))
        x = r.uniform(-1., 1., (n_ellipse,))
        y = r.uniform(-1., 1., (n_ellipse,))
        rot = r.uniform(0., 2*np.pi, (n_ellipse,))
        ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
        image = ellipsoid_phantom(space, ellipsoids)
        image -= np.min(image)
        image /= np.max(image)

        yield image


class GeneratorTorchDataset(Dataset):
    def __init__(self, generator, length, shape=None):
        self.generator = generator
        self.length = length
        self.shape = shape or (
            (None,) * self.dataset.get_num_elements_per_sample())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        arrays = next(self.generator)
        mult_elem = isinstance(arrays, tuple)
        if not mult_elem:
            arrays = (arrays,)
        tensors = []
        for arr, s in zip(arrays, self.shape):
            t = torch.from_numpy(np.asarray(arr))
            if s is not None:
                t = t.view(*s)
            tensors.append(t)
        return tuple(tensors) if mult_elem else tensors[0]


if __name__ == '__main__':
    SIZE = 32
    NUM_ANGLES = 30

    space = uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                          shape=(SIZE, SIZE), dtype=np.float32)
    geometry = parallel_beam_geometry(space, num_angles=NUM_ANGLES)
    ray_trafo = RayTransform(space, geometry, impl='astra_cpu')  # other
#    options are 'astra_cpu' and 'skimage'

    # pure ellipses generator and dataset
    ellipses_gen = ellipses_generator(ray_trafo)
    ellipses_dataset = GeneratorTorchDataset(ellipses_gen, 32000,
                                             shape=(1, 1, SIZE, SIZE))

    # pair generator and dataset
    pair_gen = pair_generator(ray_trafo)
    pair_dataset = GeneratorTorchDataset(
        pair_gen, 32000, shape=((1, NUM_ANGLES, -1), (1, 1, SIZE, SIZE)))

    (y, x) = pair_dataset[None]  # get a sample from dataset (index is ignored)
