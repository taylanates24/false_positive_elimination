from torch2trt import TRTModule
import torch
from data.dataset import pre_process_image
import argparse

class Infer:
    
    def __init__(self, checkpoint_pth) -> None:
        
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(checkpoint_pth))
        self.class_names = ['False positive', 'True positive']
        
    def classify_image(self, image):
        
        out = self.model(image)
        _, pred = torch.max(out, 1)
        
        return self.class_names[pred]
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='best_trt.pth', 
                        help='checkpoint path of tensorrt model')
    parser.add_argument('--img_path', type=str, 
                        default='dataset/images/test/8973.jpg', 
                        help='the path of input image') 
    parser.add_argument('--input_size', type=int, 
                        default=480, 
                        help='the size of the input image')  

    args = parser.parse_args()
    infer = Infer(args.checkpoint)

    image = pre_process_image(args.img_path, args.input_size).cuda()

    res = infer.classify_image(image)
    
    print(res)