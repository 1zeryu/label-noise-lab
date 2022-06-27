import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.cuda
import torch.functional as F
from random import random
from torch.autograd import Variable

# Currently there is a risk of dropping all paths...
# We should create a version that take all paths into account to make sure one stays alive
# But then keep_prob is meaningless and we have to copute/keep track of the conditional probability
class DropPath(nn.Module):
    def __init__(self, module, keep_prob=0.9):
        super(DropPath, self).__init__()
        self.module = module
        self.keep_prob = keep_prob
        self.shape = None
        self.training = True
        self.dtype = torch.FloatTensor

    def forward(self, *input):
        if self.training:
            # If we don't now the shape we run the forward path once and store the output shape
            if self.shape is None:
                temp = self.module(*input)
                self.shape = temp.size()
                if temp.data.is_cuda:
                    self.dtype = torch.cuda.FloatTensor
                del temp
            p = random()
            if p <= self.keep_prob:
                return Variable(self.dtype(self.shape).zero_())
            else:
                return self.module(*input)/self.keep_prob # Inverted scaling
        else:
            return self.module(*input)
class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class TwoSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(TwoSeparables, self).__init__()
        self.separable_0 = SeparableConv2d(in_channels, in_channels, dw_kernel, dw_stride, dw_padding, bias=bias)
        self.bn_0 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.separable_1 = SeparableConv2d(in_channels, out_channels, dw_kernel, 1, dw_padding, bias=bias)
        self.bn_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = relu(x)
        x = self.separable_0(x)
        x = self.bn_0(x)
        x = relu(x)
        x = self.separable_1(x)
        x = self.bn_1(x)
        return x

class ResizeCell0(nn.Module):
    def __init__(self, in_channels_x, in_channels_h, out_channels):
        super(ResizeCell0, self).__init__()
        self.pool_left_0 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_left_0 = nn.Conv2d(in_channels_h, out_channels//2, 1, stride=1, bias=False)

        self.pool_left_1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_left_1 = nn.Conv2d(in_channels_h, out_channels//2, 1, stride=1, bias=False)

        self.bn_left = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

        self.conv_right = nn.Conv2d(in_channels_x, out_channels, 1, stride=1, bias=False)
        self.bn_right = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x, h):
        h = relu(h)

        h_0 = self.pool_left_0(h)
        h_0 = self.conv_left_0(h_0)

        h_1 = self.pool_left_1(h)
        h_1 = self.conv_left_1(h_1)

        h = torch.cat([h_0, h_1], 1)
        h = self.bn_left(h)

        x = relu(x)
        x = self.conv_right(x)
        x = self.bn_right(x)

        return x, h

class ResizeCell1(nn.Module):
    def __init__(self, in_channels_x, in_channels_h, out_channels):
        super(ResizeCell1, self).__init__()
        self.conv_left = nn.Conv2d(in_channels_x, out_channels, 1, stride=1, bias=False)
        self.bn_left = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

        self.conv_right = nn.Conv2d(in_channels_h, out_channels, 1, stride=1, bias=False)
        self.bn_right = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x, h):
        x = relu(x)
        x = self.conv_left(x)
        x = self.bn_left(x)

        h = relu(h)
        h = self.conv_right(h)
        h = self.bn_right(h)

        return x, h


class ReductionCell(nn.Module):
    def __init__(self, in_channels_x, in_channels_h, out_channels, resize_cell=ResizeCell1, keep_prob=0.9):
        super(ReductionCell, self).__init__()

        self.resize = resize_cell(in_channels_x, in_channels_h, out_channels)

        self.comb_iter_0_left = DropPath(TwoSeparables(out_channels, out_channels, 7, 2, 3, bias=False), keep_prob)
        self.comb_iter_0_right = DropPath(TwoSeparables(out_channels, out_channels, 5, 2, 2, bias=False), keep_prob)

        self.comb_iter_1_left = DropPath(nn.MaxPool2d(3, stride=2, padding=1), keep_prob)
        self.comb_iter_1_right = DropPath(TwoSeparables(out_channels, out_channels, 7, 2, 3, bias=False), keep_prob)

        self.comb_iter_2_left = DropPath(nn.AvgPool2d(3, stride=2, padding=1), keep_prob)
        self.comb_iter_2_right = DropPath(TwoSeparables(out_channels, out_channels, 5, 2, 2, bias=False), keep_prob)

        self.comb_iter_3_left = DropPath(nn.MaxPool2d(3, stride=2, padding=1), keep_prob)
        self.comb_iter_3_right = DropPath(TwoSeparables(out_channels, out_channels, 3, 1, 1, bias=False), keep_prob)

        self.comb_iter_4_left = DropPath(nn.AvgPool2d(3, stride=1, padding=1), keep_prob)

    def forward(self, x, h):
        prev = x

        x, h = self.resize(x, h)

        comb_iter_0_left = self.comb_iter_0_left(h)
        comb_iter_0_right = self.comb_iter_0_right(x)
        comb_iter_0 = comb_iter_0_left + comb_iter_0_right

        comb_iter_1_left = self.comb_iter_1_left(x)
        comb_iter_1_right = self.comb_iter_1_right(h)
        comb_iter_1 = comb_iter_1_left + comb_iter_1_right

        comb_iter_2_left = self.comb_iter_2_left(x)
        comb_iter_2_right = self.comb_iter_2_right(h)
        x_comb_iter_2 = comb_iter_2_left + comb_iter_2_right

        comb_iter_3_left = self.comb_iter_3_left(x)
        comb_iter_3_right = self.comb_iter_3_right(comb_iter_0)
        comb_iter_3 = comb_iter_3_left + comb_iter_3_right

        comb_iter_4_left = self.comb_iter_4_left(comb_iter_0)
        comb_iter_4 = comb_iter_4_left + comb_iter_1

        return torch.cat([comb_iter_1, x_comb_iter_2, comb_iter_3, comb_iter_4], 1), prev

class NormalCell(nn.Module):
    def __init__(self, in_channels_x, in_channels_h, out_channels, resize_cell=ResizeCell1, keep_prob=0.9):
        super(NormalCell, self).__init__()
        self.adjust = resize_cell(in_channels_x, in_channels_h, out_channels)

        self.comb_iter_0_left = DropPath(TwoSeparables(out_channels, out_channels, 3, 1, 1, bias=False), keep_prob)

        self.comb_iter_1_left = DropPath(TwoSeparables(out_channels, out_channels, 3, 1, 1, bias=False), keep_prob)
        self.comb_iter_1_right = DropPath(TwoSeparables(out_channels, out_channels, 5, 1, 2, bias=False), keep_prob)

        self.comb_iter_2_left = DropPath(nn.AvgPool2d(3, stride=1, padding=1), keep_prob)

        self.comb_iter_3_left = DropPath(nn.AvgPool2d(3, stride=1, padding=1), keep_prob)
        self.comb_iter_3_h = DropPath(nn.AvgPool2d(3, stride=1, padding=1), keep_prob)

        self.comb_iter_4_left = DropPath(TwoSeparables(out_channels, out_channels, 5, 1, 2, bias=False), keep_prob)
        self.comb_iter_4_right = DropPath(TwoSeparables(out_channels, out_channels, 3, 1, 1, bias=False), keep_prob)

    def forward(self, x, h):
        prev = x

        x, h = self.adjust(x, h)

        comb_iter_0_left = self.comb_iter_0_left(x)
        comb_iter_0 = comb_iter_0_left + x

        comb_iter_1_left = self.comb_iter_1_left(h)
        comb_iter_1_right = self.comb_iter_1_right(x)
        comb_iter_1 = comb_iter_1_left + comb_iter_1_right

        comb_iter_2_left = self.comb_iter_2_left(x)
        comb_iter_2 = comb_iter_2_left + h

        comb_iter_3_left = self.comb_iter_3_left(h)
        comb_iter_3_right = self.comb_iter_3_h(h)
        comb_iter_3 = comb_iter_3_left + comb_iter_3_right

        comb_iter_4_left = self.comb_iter_4_left(h)
        comb_iter_4_right = self.comb_iter_4_right(h)
        comb_iter_4 = comb_iter_4_left + comb_iter_4_right

        return torch.cat([x, comb_iter_0, comb_iter_1, comb_iter_2, comb_iter_3, comb_iter_4], 1), prev

import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.functional import relu

class NASNet(nn.Module):
    def __init__(self, stem_filters, normals, filters, scaling, num_classes, use_aux=True, pretrained=True):
        super(NASNet, self).__init__()
        self.normals = normals
        self.use_aux = use_aux
        self.num_classes = num_classes

        self.stemcell = nn.Sequential(
            nn.Conv2d(3, stem_filters, kernel_size=3, stride=2),
            nn.BatchNorm2d(stem_filters, eps=0.001, momentum=0.1, affine=True)
        )

        self.reduction1 = ReductionCell(in_channels_x=stem_filters,
                                        in_channels_h=stem_filters,
                                        out_channels=int(filters * scaling ** (-2)),
                                        resize_cell=ResizeCell1)
        self.reduction2 = ReductionCell(in_channels_x=int(4*filters * scaling ** (-2)),
                                        in_channels_h=stem_filters,
                                        out_channels=int(filters * scaling ** (-1)),
                                        resize_cell=ResizeCell0)

        x_channels = int(4*filters * scaling ** (-1))
        h_channels = int(4*filters * scaling ** (-2))

        self.add_module('normal_block1_0',
                        NormalCell(in_channels_x=x_channels,
                                   in_channels_h=h_channels,
                                   out_channels=filters,
                                   resize_cell=ResizeCell0,
                                   keep_prob=0.9))
        # TODO: Can we do that in a cleaner way?
        h_channels = x_channels
        x_channels = 6*filters
        for i in range(normals-1):
            self.add_module('normal_block1_{}'.format(i+1),
                            NormalCell(in_channels_x=x_channels,
                                       in_channels_h=h_channels,
                                       out_channels=filters,
                                       resize_cell=ResizeCell1,
                                       keep_prob=0.9))
            h_channels = x_channels
            x_channels = 6*filters

        self.reduction3 = ReductionCell(in_channels_x=x_channels,
                                        in_channels_h=h_channels,
                                        out_channels=filters * scaling)

        h_channels = x_channels
        x_channels = 4 * filters * scaling

        self.add_module('normal_block2_0',
                        NormalCell(in_channels_x=x_channels,
                                   in_channels_h=h_channels,
                                   out_channels=filters*scaling,
                                   resize_cell=ResizeCell0,
                                   keep_prob=0.9))
        h_channels = x_channels
        x_channels = 6 * filters * scaling
        for i in range(normals - 1):
            self.add_module('normal_block2_{}'.format(i + 1),
                            NormalCell(in_channels_x=x_channels,
                                       in_channels_h=h_channels,
                                       out_channels=filters*scaling,
                                       resize_cell=ResizeCell1, keep_prob=0.9))
            h_channels = x_channels
            x_channels = 6 * filters * scaling

        self.reduction4 = ReductionCell(in_channels_x=x_channels,
                                        in_channels_h=h_channels,
                                        out_channels=filters * scaling ** 2)

        h_channels = x_channels
        x_channels = 4 * filters * scaling ** 2

        self.add_module('normal_block3_0',
                        NormalCell(in_channels_x=x_channels,
                                   in_channels_h=h_channels,
                                   out_channels=filters * scaling ** 2,
                                   resize_cell=ResizeCell0, keep_prob=0.9))
        h_channels = x_channels
        x_channels = 6 * filters * scaling ** 2
        for i in range(normals - 1):
            self.add_module('normal_block3_{}'.format(i + 1),
                            NormalCell(in_channels_x=x_channels,
                                       in_channels_h=h_channels,
                                       out_channels=filters * scaling ** 2,
                                       resize_cell=ResizeCell1,
                                       keep_prob=0.9))
            h_channels = x_channels
            x_channels = 6 * filters * scaling ** 2

        self.avg_pool_0 = nn.AvgPool2d(1, stride=1, padding=0)
        self.dropout_0 = nn.Dropout()
        self.fc = nn.Linear(x_channels, self.num_classes)

    def features(self, x):
        x = self.stemcell(x)

        x, h = self.reduction1(x, x)
        x, h = self.reduction2(x, h)

        for i in range(self.normals):
            x, h = self._modules['normal_block1_{}'.format(i)](x, h)

        x, h = self.reduction3(x, h)

        for i in range(self.normals):
            x, h = self._modules['normal_block2_{}'.format(i)](x, h)

        # Should we check for training or not ?
        if self.use_aux and self.training:
            x_aux = x

        x, h = self.reduction4(x, h)

        for i in range(self.normals):
            x, h = self._modules['normal_block3_{}'.format(i)](x, h)

        if self.use_aux and self.training:
            return x, x_aux
        else:
            return x

    def classifier(self, x):
        x = relu(x)
        x = self.avg_pool_0(x)
        x = x.view(-1, self.fc.in_features)
        x = self.dropout_0(x)
        x = self.fc(x)
        return x

    def aux_classifier(self, x):
        x = relu(x)
        x = self.avg_pool_0(x)
        x = x.view(-1, self.fc.in_features)
        x = self.dropout_0(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        if self.use_aux:
            x, x_b = self.features(x)
            x = self.classifier(x)
            x_b = self.aux_classifier(x_b)
            return x, x_b
        else:
            x = self.features(x)
            x = self.classifier(x)
            return x


def nasnetmobile(num_classes=1000, pretrained=False,use_aux=True):
    return NASNet(32, 4, 44, 2, num_classes=num_classes, use_aux=use_aux, pretrained=pretrained)

def nasnetlarge(num_classes=1000, pretrained=False,use_aux=True):
    return NASNet(96, 6, 168, 2, num_classes=num_classes, use_aux=use_aux, pretrained=pretrained)