import numbers
import numpy as np
import torch
from skimage import color


# HED Jitter
# Adapted from https://github.com/gatsby2016/Augmentation-PyTorch-Transforms

class HEDJitter(object):
    """Randomly perturbe the HED color space value an RGB image.
    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin and DAB stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         betti is chosen from a uniform distribution [-theta, theta]
         the jitter formula is **s' = \alpha * s + \betti**
    """
    def __init__(self, theta=0.): # HED_light: theta=0.05; HED_strong: theta=0.2
        assert isinstance(theta, numbers.Number), "theta should be a single number."
        self.theta = theta
        self.alpha = np.random.uniform(1-theta, 1+theta, (1, 3))
        self.betti = np.random.uniform(-theta, theta, (1, 3))

    @staticmethod
    def adjust_HED(img, alpha, betti):
        if img.shape[0] == 3:
          move_axis = True
          img = torch.moveaxis(img, 0, 2)
        else:
          move_axis = False

        s = np.reshape(color.rgb2hed(img), (-1, 3))
        ns = alpha * s + betti  # perturbations on HED color space
        nimg = color.hed2rgb(np.reshape(ns, img.shape))

        imin = nimg.min()
        imax = nimg.max()
        if imax - imin > 0:
          rsimg = ((nimg - imin) / (imax - imin))  # rescale to [0,255]
        else:
          rsimg = nimg

        if move_axis:
          rsimg = torch.moveaxis(torch.tensor(rsimg, dtype=torch.float32), 2, 0)

        return rsimg

    def __call__(self, img):
        return self.adjust_HED(img, self.alpha, self.betti)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'theta={0}'.format(self.theta)
        format_string += ',alpha={0}'.format(self.alpha)
        format_string += ',betti={0}'.format(self.betti)
        return format_string
