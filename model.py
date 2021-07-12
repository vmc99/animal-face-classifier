import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import math
import gc
import streamlit as st


#Enable garbage collection
gc.enable()

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)] 
    if pool :
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # in : 3 x 224 x 224
        self.conv1 = conv_block(in_channels, 32) # 32 x 224 x 224
        self.conv2 = conv_block(32, 64, pool=True) # 64 x 112 x 112
        self.res1 = nn.Sequential(conv_block(64, 64),
                                  conv_block(64, 64)) # 64 x 112 x 112

        self.conv3 = conv_block(64, 128, pool=True) # 128 x 56 x 56
        self.conv4 = conv_block(128, 256, pool=True) # 256 x 28 x 28
        self.res2 = nn.Sequential(conv_block(256, 256),
                                  conv_block(256, 256)) # 256 x 28 x 28

        self.conv5 = conv_block(256, 512, pool=True) # 512 x 14 x 14
        self.conv6 = conv_block(512, 1024, pool=True) # 1024 x 7 x 7
        self.res3 = nn.Sequential(conv_block(1024, 1024),
                                  conv_block(1024, 1024)) # 1024 x 7 x 7

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1),  # 1024 x 1 x 1 
                                        nn.Flatten(),    # 1024*1*1 
                                        nn.Linear(1024, 512),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.classifier(out)

        return out


# def get_default_device():
#     """Pick GPU if available, else CPU"""
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')


def transform_image(image):
    stats = ((0.5025, 0.4600, 0.3992), (0.2552, 0.2455, 0.2501))
    my_transforms = T.Compose([
                        T.Resize(225),
                        T.CenterCrop(224), 
                        T.ToTensor(),
                        T.Normalize(*stats)])
    return my_transforms(image)


@st.cache
def initiate_model():

    # Initiate model
    in_channels = 3
    num_classes = 3
    model = ResNet(in_channels, num_classes)
    device = torch.device('cpu')
    PATH = 'animal-face-detection_new.pth'
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    return model






def predict_image(img):
    
    classes = ['Cat', 'Dog', 'Wild']
    # Convert to a batch of 1
    xb = img.unsqueeze(0)

    model = initiate_model()

    # Get predictions from model
    yb = model(xb)
    # apply softamx
    yb_soft = F.softmax(yb, dim=1)
    # Pick index with highest probability
    confidence , preds  = torch.max(yb_soft, dim=1)
    gc.collect()
    # Retrieve the class label and probability
    return classes[preds[0].item()], math.trunc(confidence.item()*100)






