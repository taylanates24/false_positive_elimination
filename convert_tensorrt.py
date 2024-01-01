from model import SceneClassifier
import torch
from torch2trt import torch2trt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='best.ckpt', 
                        help='checkpoint path of torch model')
    parser.add_argument('--num_classes', type=int, 
                        default=2, 
                        help='number of classes') 
    parser.add_argument('--input_size', type=int, 
                        default=480, 
                        help='the size of the input image') 
    parser.add_argument('--save_path', type=str, 
                        default='best_trt.pth', 
                        help='the path of tensorrt model')  
    
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    model_state_dict = checkpoint['model_state_dict']
    model = SceneClassifier(pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(model_state_dict)

    print('The pretrained model is loaded.')

    image = torch.randn(1, 3, args.input_size, args.input_size).cuda()
    model.eval().cuda()

    model_trt = torch2trt(model, [image])
    print('The model is converting to TensorRT.')
    torch.save(model_trt.state_dict(), args.save_path)
    print('TensorRT model is saved.')