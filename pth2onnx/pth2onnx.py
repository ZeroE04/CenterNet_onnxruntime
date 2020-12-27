# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:20:07 2020

@author: Lim
"""

import torch
import torch.onnx as onnx
from dlanet import DlaNet
# from dlanet_dcn import DlaNet
from torch.onnx import OperatorExportTypes

model = DlaNet(34,plot=True)

model.load_state_dict(torch.load('model//r_dla_34.pth'))
model.eval()
# model.cuda()
# inputs = torch.zeros([1, 3, 512, 512]).cuda()
inputs = torch.zeros([1, 3, 512, 512])
onnx.export(model, inputs, "model//r_dla_34.onnx", verbose=True,operator_export_type=OperatorExportTypes.ONNX)
    