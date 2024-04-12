import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import gaussian_blur

import numpy as np

# Code largely adapted from https://github.com/ouyangzhibo/Image_Foveation_Python.git and rewritten to use PyTorch

def _Gaussian_pyramid(im, kernel_size=5, sigmas=[], num_levels=6):
    A = torch.clone(im)
    pyramids = [A]
    
    _, height_original, width_original = im.shape
    
    # Apply Gaussian blur to every level (except 0) and downsample by factor of 2
    for i in range(1, num_levels):
        A = gaussian_blur(A, kernel_size=kernel_size, sigma=sigmas[i])

        _, height, width = A.shape
        A = Resize((int(height/2), int(width/2)))(A)

        pyramids.append(A)

    # Upsample every level (except 0) by factor of 2 until reaching original size
    for i in range(1, num_levels):
        A = pyramids[i]
        for j in range(i):
            if j < i-1:
                new_size = (A.shape[1]*2, A.shape[2]*2)
            else:
                new_size = (height_original, width_original)

            A = Resize(new_size)(A)
            # P = gaussian_blur(A, kernel_size=kernel_size, sigma=sigmas[i])
        pyramids[i] = A

    return pyramids

def foveate_image(im, fovea_pos):
    num_levels = 6
    # sigmas = [0.248, 0.124, 0.056, 0.0267, 0.0131, 0.00654]  # standard deviation of the Gaussian distribution
    sigmas = [0.248]  # standard deviation of the Gaussian distribution
    if len(sigmas) == 1:
        sigmas = np.repeat(sigmas[0], num_levels)

    p = 3.5  # number of pixels a person can see in a degree of visual angle
    alpha = 2.5  # half-height angle: when θ(x, y) = α, the image will become only half the resolution of the center of attention

    # Fixation
    x_focus, y_focus = fovea_pos
    
    GP = _Gaussian_pyramid(im, num_levels=num_levels, sigmas=sigmas)

    x = torch.arange(0, im.shape[2], 1, dtype=torch.float32)
    y = torch.arange(0, im.shape[1], 1, dtype=torch.float32)
    x_2d, y_2d = torch.meshgrid(x, y, indexing='ij')

    # Map image coordinates to visual angles
    theta = torch.sqrt((x_2d - x_focus) ** 2 + (y_2d - y_focus) ** 2) / p

    # Resolution map
    R = alpha / (theta + alpha)  # shape [width, height]; values between 1 (at center of fovea) and 0
    
    # Transfer functions
    Ts = []
    for i in range(1, num_levels):
        Ts.append(torch.exp(-((2 ** (i-3)) * R / sigmas[i-1]) ** 2 * 0.5))
    Ts.append(torch.zeros_like(R))  # equal to 0 for i == num_levels
    
    # Bandwidths where Ti(omegai) = 0.5
    omegas = torch.zeros(num_levels)
    for i in range(1, num_levels):
        omegas[i - 1] = (sigmas[i - 1] * torch.sqrt(torch.log(torch.tensor(2)))) / (2 ** (i - 3) * torch.sqrt(torch.tensor(0.5)))
    omegas[-1] = torch.tensor(0)

    # Normalize bandwidths
    omegas_norm = []
    omega_max = torch.max(omegas)
    omega_min = torch.min(omegas)
    for omega in omegas:
        omegas_norm.append((omega - omega_min) / (omega_max - omega_min))

    # Layer numbers
    layer_numbers = torch.zeros_like(R)
    for i in range(1, num_levels):
        i0 = torch.logical_and(R >= omegas_norm[i], R <= omegas_norm[i - 1])
        layer_numbers[i0] = i
    layer_numbers = layer_numbers.type(torch.IntTensor)

    # Blending functions
    Bs = []
    for i in range(1, num_levels):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))  # 1e-5 to avoid division by 0
    
    # Blending coefficients
    Ms = torch.zeros((num_levels, R.shape[0], R.shape[1]))
    for i in range(1, num_levels + 1):
        index = layer_numbers == i
        if torch.sum(index) > 0:
            if i == 1:
                Ms[i-1][index] = 1
            else:
                Ms[i-1][index] = 1 - Bs[i-1][index]

        index = layer_numbers - 1 == i
        if torch.sum(index) > 0:
            Ms[i-1][index] = Bs[i][index]

    # Generate foveated image
    im_fov = torch.zeros_like(GP[0], dtype=torch.float32)
    # Linear combination of Ms and As
    for index, (M, A) in enumerate(zip(Ms, GP)):
        for color in range(im.shape[0]):
            im_fov[color, :, :] += torch.multiply(M.T, A[color, :, :])
    # im_fov = im_fov.type(torch.IntTensor)

    im_fov = torch.clamp(im_fov, 0, 1)

    # print('Num full-res pixels', torch.sum(Ms[0] == 1))

    fovea_mask = torch.where((Ms[0] == 1), 0, 1)  # 0 where fovea is, 1 in periphery

    return im_fov, fovea_mask
