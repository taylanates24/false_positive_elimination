import imgaug.augmenters as iaa
import numpy as np
import random


class Augmentations:
    
    def __init__(self, opt) -> None:

        augmentations = []

        if opt['scale'] and random.random() < opt['scale'][-1]:
            augmentations.append(iaa.Affine(scale=opt['scale'][:2]))
        if opt['brightness'] and random.random() < opt['brightness'][-1]:   
            augmentations.append(iaa.AddToBrightness(opt['brightness'][:2]))
        if opt['saturation'] and random.random() < opt['saturation'][-1]:   
            augmentations.append(iaa.AddToSaturation(opt['saturation'][:2]))
        if opt['hue'] and random.random() < opt['hue'][-1]:   
            augmentations.append(iaa.AddToHue(opt['hue'][:2]))
        if opt['add_grayscale'] and random.random() < opt['add_grayscale'][-1]:   
            augmentations.append(iaa.Grayscale(alpha=opt['add_grayscale'][:2]))
        if opt['motion_blur'] and random.random() < opt['motion_blur'][-1]:   
            augmentations.append(iaa.MotionBlur(k=opt['motion_blur'][:2]))
        if opt['translate'] and random.random() < opt['translate'][-1]:   
            augmentations.append(iaa.Affine(translate_percent={"x": opt['translate'][0], "y": opt['translate'][1]}))
        if opt['rotate'] and random.random() < opt['rotate'][-1]:   
            augmentations.append(iaa.Affine(rotate=opt['rotate'][:2]))
        if opt['shear'] and random.random() < opt['shear'][-1]:   
            augmentations.append(iaa.Affine(shear=opt['shear'][:2]))
        if opt['contrast'] and random.random() < opt['contrast'][-1]:
            augmentations.append(iaa.LinearContrast(opt['contrast'][:2]))

        if len(augmentations):
            self.seq = iaa.SomeOf(n=5, children=augmentations, random_order=True)
            self.fliplr = iaa.Sequential([iaa.Fliplr(opt['fliplr'][0])])
    def __call__(self, image):

        image = self.seq(image=image.astype(np.uint8))
        image = self.fliplr.augment_image(image)
        return image