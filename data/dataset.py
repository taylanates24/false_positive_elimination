from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import cv2
from data.utils import letter_box, load_image


class FPDataset(Dataset):
    
    def __init__(self, image_path, image_size=64, augment=False, augmentations=None) -> None:
        super(FPDataset, self).__init__()
        self.image_path = image_path
        self.image_size = image_size
        self.augment = augment
        self.augmentations = augmentations

        self.image_paths = []

        for image_cls in os.listdir(os.path.join(self.image_path)):

            for img_name in os.listdir(os.path.join(self.image_path,image_cls)):
                
                self.image_paths.append(os.path.join(self.image_path,image_cls, img_name))

                
        self.image_paths = sorted(self.image_paths)

        self.transform = transforms.Compose([
            transforms.ToTensor()
            ])

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image_name = image_path.split('/')[-1]
        image = load_image(image_path=image_path, image_size=self.image_size)

        if self.augmentations is not None:
            
            for augment in self.augmentations:
                
                image = augment(image)


        if image.shape[0] != image.shape[1]:
            
            image = letter_box(image=image, size=self.image_size)
            
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, self.image_size)

        image = self.transform(image.astype('float32')) / 255
        label = int(image_name.split('_')[0])
        
        if label == 9:
            label = 0
            
        return image, label




