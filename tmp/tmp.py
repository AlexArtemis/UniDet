# -*- coding: utf-8 -*-
# @Author  : leizehua

import torch.utils.model_zoo as model_zoo
import torch.onnx
from torch import nn
from torchvision import models
from torchsummary import summary

model = MyModel()
model = torch.load('../model_zoo/Unified_learned_OCI_RS200_6x.pth').eval().cuda()
# model.eval()
summary(model, input_size=(3, 1080, 1920))
