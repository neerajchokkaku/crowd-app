import torch
import torch.nn as nn

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, batch_norm=True)
        self.output_layer = nn.Conv2d(512, 1, kernel_size=1)

        if load_weights:
            checkpoint = torch.load('csrnet.pth', map_location='cpu')
            if 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']
            self.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)