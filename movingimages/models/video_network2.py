import torch
import torch.nn as nn
from movingimages.models.gateshift import GSF, Tsm
from movingimages.ops.basic_ops import ConsensusModule
import torchvision.models as models
# Basic Block for ResNet-18 (consists of two Convolutional layers with Batch Normalization)

class VideoResnet152(nn.Module):
    def __init__(self, pretrained=True, num_classes=100, num_segments=8):
        super(VideoResnet152, self).__init__()
        self.resnet = models.resnet152(pretrained)
        self.num_classes = num_classes
        self.num_segments = num_segments

        self.shift = GSF

        for i in range(36):
            self.resnet.layer3[i].conv3= nn.Sequential(
                self.resnet.layer3[i].conv3,
                self.shift(fPlane=self.resnet.layer3[i].conv3.in_channels, num_segments=num_segments)  # Add your attention module here
            )
        self.resnet.layer4[0].conv3 = nn.Sequential(
            self.resnet.layer4[0].conv3 ,
            self.shift(fPlane=self.resnet.layer4[0].conv3.in_channels, num_segments=num_segments) # Add your attention module here
        )
        self.resnet.layer4[1].conv3 = nn.Sequential(
            self.resnet.layer4[1].conv3 ,
            self.shift(fPlane=self.resnet.layer4[1].conv3.in_channels, num_segments=num_segments) # Add your attention module here
        )

        # todo batchnorm prima o dopo di shift?

        # Modify the classifier layer for the new number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes))

        self.consensus = ConsensusModule('avg')

    def forward(self, input):
        x = input.view((-1, 3) + input.size()[-2:])
        out = self.resnet(x)
        base_out_logits = out.view((-1, self.num_segments) + out.size()[1:])
        output = self.consensus(base_out_logits)
        return output

if __name__=='__main__':

    input = torch.rand((8, 3*8, 64, 64))
    model = VideoResnet152(pretrained=True, num_classes=100, num_segments=8)
    print(model)
    output = model(input)
    print(output.shape)