import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.snconv_qurey = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.snconv_key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.snconv_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.snconv_attention = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.snmaxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps [B  C  W  H]
            returns :
                attention: [B  C  N] (N is Width*Height)
                out : input feature + (value @ attention)
        """
        b, c, h, w = x.size()

        qurey = self.snconv_qurey(x)
        qurey = qurey.view(-1, c // 8, h * w)

        key = self.snconv_key(x)
        key = self.snmaxpool(key)
        key = key.view(-1, c // 8, h * w // 4)

        attention = self.softmax(torch.bmm(qurey.permute(0, 2, 1), key))

        value = self.snconv_value(x)
        value = self.snmaxpool(value)
        value = value.view(-1, c // 2, h * w // 4)

        a = torch.bmm(value, attention.permute(0, 2, 1))
        a = a.view(-1, c // 2, h, w)
        a = self.snconv_attention(a)

        out = x + self.sigma * a
        return out
