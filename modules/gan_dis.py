import torchvision.models as vision_model
import torch.nn as nn
import torch
from models import TrainableModel

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class GanDisNet(TrainableModel):
    def __init__(self):
        super(GanDisNet, self).__init__()
        self.size = 224
        self.backbone = vision_model.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features,1)
        self.apply(weight_init)

    def forward(self, x):
        x = nn.functional.interpolate(x,size=self.size, mode='bilinear',align_corners=True)
        x = self.backbone(x)
        return x

    def step(self, loss, train=True, retain_graph=False):

        self.zero_grad()
        self.optimizer.zero_grad()
        self.train(train)
        self.zero_grad()
        self.optimizer.zero_grad()

        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        self.zero_grad()
        self.optimizer.zero_grad()