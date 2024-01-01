from model import SceneClassifier
import torch
import argparse
from pytorch_lightning import LightningModule


class SimpleModel(LightningModule):
    def __init__(self, model_state_dict, num_classes):
        super().__init__()
        self.model = SceneClassifier(pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(model_state_dict)

    def forward(self, x):
        return self.model.forward(x)
    

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
                        default='best.onnx', 
                        help='the path of tensorrt model') 
    

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    model_state_dict = checkpoint['model_state_dict']
    model = SimpleModel(model_state_dict=model_state_dict, num_classes=args.num_classes)

    image = torch.randn(1, 3, args.input_size, args.input_size).cuda()

    model.to_onnx(args.save_path, image, export_params=True)