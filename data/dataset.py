import os
import math
import random
from itertools import product

import torch
from torchvision import transforms

from data.preprocess import foveate_image


transform_color = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.8,
                                contrast=0.8,
                                saturation=0.8,
                                hue=0.2)
    ], p=0.8),
])


class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, data, masking_mode='None', masking_ratio=0.5, blur=False, random_crop=False, color_jitter=False, segment=False, segment_path=None, remove_missing_segments=False, attention=False, seed=0):
        """Generates (masked) samples from the given input data. Returns masked-image, target-image, target-label, mask.

        Args:
            data (torchvision dataset): Input data used to generate (masked) samples from.
            masking_mode (string, optional): Masking paradigm and augmentations. Allowed values are 'periphery' (masked periphery), 'random_patches' (randomly masked patches), 'foveate' (bio-inspired foveation) or 'None'. Defaults to 'None'.
            masking_ratio (float, optional): Ratio of pixels to mask (ranging from 0-1 where 0 means no pixels are masked and 1 means all pixels are masked). Defaults to 0.5.
            blur (bool, optional): Whether or not to apply Gaussian blur to masked regions. Defaults to False.
            random_crop (bool, optional): Whether or not to apply random resized cropping (scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)) to sample. Defaults to False.
            color_jitter (bool, optional): Whether or not to apply random color jitter to sample. Defaults to False.
            segment (bool, optional): Whether or not to include presegmentation in returned mask based on files at segment_path. Defaults to False.
            segment_path (string (path), optional): Path to directory containing .pt files with one presegmentation mask per image in input data. Defaults to None.
            remove_missing_segments (bool, optional): Whether to exclude images whose masks are not present at segment_path. If False, uniform mask of ones will be used in case presegmentations are missing. Defaults to False.
            attention (bool, optional): Whether or not to include a second high sampling-frequency input circle in masked sample. Position of attention circle is random if segment is False and on a random pixel with the highest segmentation-confidence otherwise. Defaults to False.
            seed (int, optional): Random seed used. Defaults to 0.
        """
        self.data = list(data)
        self.masking_mode = masking_mode
        self.masking_ratio = masking_ratio
        self.blur = blur
        self.random_crop = random_crop
        self.color_jitter = color_jitter
        self.segment = segment
        if segment_path and not os.path.exists(segment_path):
            raise Exception('Segmentation masks were not found because {} does not exist'.format(segment_path))
        self.segment_path = segment_path
        self.remove_missing_segments = remove_missing_segments
        self.attention = attention

        self.seed = seed
        random.seed(seed)

        if self.segment and self.remove_missing_segments:
            self.segment_masks = []
            removed_counter = 0
            initial_length = len(self.data)
            for i in range(initial_length - 1, -1, -1):
                if not os.path.exists(self.segment_path + 'segmentation_{}.pt'.format(i)):
                    removed_counter += 1
                    del self.data[i]
                else:
                    self.segment_masks.append(torch.load(self.segment_path + 'segmentation_{}.pt'.format(i)))
            self.segment_masks.reverse()

            print('Removed {} images'.format(removed_counter))
            print('{} remaining images'.format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index][0]  # [0] for datasets with shape (image, label)
        label = self.data[index][1]
        target = sample.copy()

        sample = transforms.ToTensor()(sample)
        target = transforms.ToTensor()(target)

        if self.segment:
            if self.remove_missing_segments:
                try:
                    segment_mask = self.segment_masks[index]
                except:
                    raise Exception((index, len(self.segment_masks)))
                segment_mask = segment_mask.unsqueeze(0).repeat(3, 1, 1)
            else:
                if os.path.exists(self.segment_path + 'segmentation_{}.pt'.format(index)):
                    segment_mask = torch.load(self.segment_path + 'segmentation_{}.pt'.format(index))
                    segment_mask = segment_mask.unsqueeze(0).repeat(3, 1, 1)
                else:
                    segment_mask = torch.ones_like(sample)

        if self.random_crop:
            top1, left1, height1, width1 = transforms.RandomResizedCrop.get_params(sample, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))
            sample = transforms.functional.resized_crop(sample, top1, left1, height1, width1, size=(96, 96))
            target = torch.clone(sample)

            if self.segment:
                segment_mask = transforms.functional.resized_crop(segment_mask, top1, left1, height1, width1, size=(96, 96))

        if self.color_jitter:
            sample = transform_color(sample)
            target = torch.clone(sample)

        if self.blur:
            blurred_sample = transforms.GaussianBlur(kernel_size=33, sigma=100)(sample)

        if self.masking_mode == 'foveate':
            sample, mask = foveate_image(sample, (int(sample.shape[2]/2), int(sample.shape[1]/2)))
            mask = mask.unsqueeze(0).repeat(3, 1, 1)

        elif self.masking_mode == 'periphery':
            mask = torch.ones_like(sample)

            patch_radius = int(math.sqrt(((sample.shape[1] * sample.shape[2]) * (1 - self.masking_ratio)) / math.pi))

            center_x = sample.shape[2] // 2
            center_y = sample.shape[1] // 2

            if self.attention:
                if self.segment:
                    # All indices containing the highest confidence value
                    max_confidence_index = (segment_mask[0]==torch.max(segment_mask[0])).nonzero()
                    max_confidence_index = max_confidence_index[random.randint(0, (max_confidence_index.shape[0] - 1))]  # Pick a random one

                    center_x2 = max_confidence_index[0]
                    center_y2 = max_confidence_index[1]
                else:
                    center_x2 = torch.randint(low=0, high=sample.shape[1], size=(1,))
                    center_y2 = torch.randint(low=0, high=sample.shape[2], size=(1,))

            # Create a meshgrid for the x and y coordinates
            x_coords, y_coords = torch.meshgrid([torch.arange(sample.shape[1]), torch.arange(sample.shape[2])], indexing='ij')

            # Create a circular mask using the Euclidean distance from the center
            temp_mask = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) > patch_radius ** 2
            if self.attention:
                temp_mask2 = ((x_coords - center_x2) ** 2 + (y_coords - center_y2) ** 2) > patch_radius ** 2

            mask = mask * (temp_mask.float().unsqueeze(0))
            if self.attention:
                mask2 = torch.ones_like(sample)
                mask2 = mask2 * (temp_mask2.float().unsqueeze(0))

                attention_circle = torch.where(mask2 == 0)
                mask[attention_circle] = 0

        elif self.masking_mode == 'random_patches':
            mask = torch.zeros_like(sample)

            patch_indices = list(product(range(12), repeat=2))  # divide image into 144 8x8 patches

            # random.seed(self.seed)  # for static mask
            random.shuffle(patch_indices)

            masked_patches = patch_indices[:int(144 * self.masking_ratio)]  # mask out approx. masking_ratio patches

            for patch in masked_patches:
                mask[:, patch[0]*8:8 + (patch[0]*8), patch[1]*8:8 + (patch[1]*8)] = 1

        elif self.masking_mode == 'None':
            mask = torch.ones_like(sample)

        else:
            raise Exception('Unknown input for masking_mode {}. Supported masking_modes are "periphery", "random_patches", "foveate", and "None".'.format(self.masking_mode))

        if self.blur:
            sample = torch.where(mask == 1, blurred_sample, sample)

        if self.masking_mode != 'None' and self.masking_mode != 'foveate':
            average_color = torch.mean(sample, dim = [1,2]).unsqueeze(1).unsqueeze(2)
            average_color_sample = torch.ones_like(sample) * average_color

            sample = torch.where(mask == 1, average_color_sample, sample)

        if self.segment:
            mask = mask * segment_mask

        return sample, target, label, mask
