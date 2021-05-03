import numpy as np
import torch
from torch.nn import funtional as F
import functools.reduce

def image_var(img, ksize, outplanes=1, inplanes=3):
    """

    :param img:
    :param ksize:
    :param outplanes:
    :param inplanes:
    :return:
    [bs, 1, h, w]
    0 -- most textured
    15 -- most textureless
    """
    thresh_list = [0.000, 0.001, 0.00163789, 0.0026827,
                   0.00439397, 0.00719686, 0.01178769, 0.01930698,
                   0.03162278, 0.05179475, 0.08483429, 0.13894955,
                   0.22758459, 0.37275937, 0.61054023, 1.]

    w = torch.ones((outplanes, inplanes, ksize, ksize)).to(img) / (3 * ksize * ksize)
    mean_local = F.conv2d(input=img, weight=w, padding=ksize // 2)

    mean_square_local = F.conv2d(input=img ** 2, weight=w, padding=ksize // 2)
    std_local = (mean_square_local - mean_local ** 2) * (3 * ksize ** 2) / (3 * ksize ** 2 - 1)

    epsilon = 1e-6

    img_var = (std_local - std_local.min()) / (std_local.max() - std_local.min())

    label = functools.reduce(lambda a, b: a + b, [img_var < t for t in thresh_list])
    label = label.long()

    return label

