import torch.nn as nn
import timm


class FPClassifier(nn.Module):
    
    def __init__(self, model_name='tf_efficientnet_lite2', pretrained=True, num_classes=3):

        super(FPClassifier, self).__init__()
        #avail_pretrained_models = timm.list_models()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, image):
        
        return self.model(image)
    