import cv2
import numpy as np 
import torchvision.transforms as transforms


def load_image(image_path, image_size):

    image = cv2.imread(image_path)

    height, width = image.shape[:2]
    ratio = image_size / max(height, width)            

    if ratio != 1:

        image = cv2.resize(image, (int(width*ratio), int(height*ratio)), interpolation=cv2.INTER_CUBIC)
    
    return image


def letter_box(image, size):
    
    box = np.full([size, size, image.shape[2]], 127)
    h, w = image.shape[:2]
    h_diff = size - h
    w_diff = size - w
    
    if h_diff > w_diff:
        
        box[int(h_diff/2):int(image.shape[0]+h_diff/2), :image.shape[1], :] = image

    else:
        
        box[:image.shape[0], int(w_diff/2):int(image.shape[1]+w_diff/2), :] = image
    
    return box


def pre_process_image(image_path, image_size):
    
    transform = transforms.Compose([
        transforms.ToTensor()])
    image = load_image(image_path=image_path, image_size=image_size)

    if image.shape[0] != image.shape[1]:
        
        image = letter_box(image=image, size=image_size)
    
    image = transform(image.astype('float32')) / 255
    
    return image.unsqueeze(0)