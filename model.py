import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class YoloLayer(nn.Module):
    def __init__(self, anchors, anchorsMask, classesNum, fullImageSize):
        super(YoloLayer, self).__init__()
        anchorsMask = [int(a) for a in anchorsMask.split(',')]
        anchors = [int(a) for a in anchors.split(',')]
        self.anchors = torch.Tensor([[anchors[2*m],anchors[2*m+1]] for m in anchorsMask])
        self.anchorsNum = len(self.anchors)
        self.classesNum = int(classesNum)
        self.fullImageSize = fullImageSize

    def forward(self, x, training):
        b,_, h,w = x.shape
        # reshape (b,255,h,w) to (b,3,h,w,85) since we predict 3 anchor boxes for each grid cell
        #  and 85 properties (80 class score, 2 position, 2 size and 1 objectness) for each prediction
        x = x.reshape(b, self.anchorsNum, self.classesNum + 5, h, w).permute(0,1,3,4,2)

        x[..., 4:] = x[..., 4:].sigmoid()

        if training:
            return x

        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij')
        grid = torch.stack((xv, yv), 2).float().to(x.device).view((1,1,h,w,2))
        anchorGrid = self.anchors.view((1,self.anchorsNum,1,1, 2)).to(x.device)

        x[..., 0:2] = (x[..., 0:2].sigmoid() + grid) / w
        x[..., 2:4] = torch.exp(x[..., 2:4]) * anchorGrid / 416
        return x


class CustomModel(nn.Module):
    def __init__(self, netProperties, layerProperties):
        super(CustomModel, self).__init__()
        floatParams = ['learning_rate', 'decay', 'momentum']
        intParams = ['width', 'height', 'channels', 'batch','subdivisions']
        self.hyperparams = {key: float(val) for key,val in netProperties.items() if key in floatParams+intParams}
        for p in intParams: self.hyperparams[p] = int(self.hyperparams[p])
        self.layerProperties = layerProperties
        self.layers = []
        self.yoloLayers = []
        self.trainableModules = nn.ModuleList() # contains the trainable layers
        self.layersToBeSaved = []
        self.createLayers()

    def createLayers(self):
        layerOutputChannels = []
        for layerProp in self.layerProperties:
            if layerProp['type'] == 'convolutional':
                # layerStack holds the conv layer, and additionally batch norm and/or activation
                layerStack = {}
                convOutChannels = int(layerProp['filters'])
                paddingValue = (int(layerProp['size'])-1)//2
                layerStack['conv'] = nn.Conv2d(in_channels=layerOutputChannels[-1] if 0<len(layerOutputChannels) else int(self.hyperparams['channels']),
                                                out_channels=convOutChannels,
                                                kernel_size=int(layerProp['size']),
                                                stride=int(layerProp['stride']),
                                                padding=tuple(paddingValue for _ in range(2)),
                                                bias=(0==int(layerProp.get('batch_normalize', 0)))
                                                )
                if 1 == int(layerProp.get('batch_normalize', 0)):
                    layerStack['batch_normalize'] = nn.BatchNorm2d(convOutChannels)
                    self.trainableModules.append(layerStack['batch_normalize'])  # bn layer should learn
                if layerProp['activation'] == 'leaky':
                    layerStack['leaky'] = nn.LeakyReLU(0.1)
                self.layers.append({'type': 'conv', 'functions': layerStack})
                layerOutputChannels.append(convOutChannels)
                self.trainableModules.append(layerStack['conv'])  # conv layer should learn
            elif layerProp['type'] == 'upsample':
                self.layers.append({'type': 'upsample', 'fun': nn.Upsample(scale_factor=int(layerProp['stride']), mode='nearest')})
                layerOutputChannels.append(layerOutputChannels[-1])
            elif layerProp['type'] == 'shortcut':
                shortcutFrom = len(self.layers)+int(layerProp['from'])  # adding since 'from' is negative
                self.layersToBeSaved.append(shortcutFrom)
                self.layers.append({'type': 'shortcut', 'from': shortcutFrom})
                layerOutputChannels.append(layerOutputChannels[shortcutFrom])
            elif layerProp['type'] == 'route':
                routeLayers = [int(l) if int(l) > 0 else len(self.layers)+int(l) for l in layerProp['layers'].split(',')]
                self.layersToBeSaved += routeLayers
                self.layers.append({'type': 'route', 'layers': routeLayers})
                layerOutputChannels.append(sum([layerOutputChannels[layerIdx] for layerIdx in routeLayers]))
            elif layerProp['type'] == 'yolo':
                self.layers.append({'type': 'yolo', 'fun': YoloLayer(layerProp['anchors'], layerProp['mask'], layerProp['classes'], self.hyperparams['width'])})
                self.yoloLayers.append(self.layers[-1]['fun'])
                layerOutputChannels.append(0)

    def forward(self, x):
        layerOutputs = {}
        netOutputs = []
        for i, layer in enumerate(self.layers):
            if layer['type'] == 'conv':
                for _,f in layer['functions'].items():
                    x = f(x)
            elif layer['type'] == 'yolo':
                x = layer['fun'](x, self.training)
                netOutputs.append(x)
            elif layer['type'] == 'upsample':
                x = layer['fun'](x)
            elif layer['type'] == 'shortcut':
                x = x + layerOutputs[layer['from']]
            elif layer['type'] == 'route':
                if 1 < len(layer['layers']):
                    x = torch.cat(tuple(layerOutputs[layerIdx] for layerIdx in layer['layers']), 1)
                else:
                    x = layerOutputs[layer['layers'][0]]

            if i in self.layersToBeSaved:
                layerOutputs[i] = x

        return netOutputs

    def loadBackboneWeights(self, weightsPath, device):
        with open(weightsPath, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        try:
            cutoff = int(weightsPath.split(".")[-1])
        except ValueError:
            cutoff = -1

        ptr = 0
        for i, layer in enumerate(self.layers):
            if i == cutoff:
                break
            if layer['type'] != 'conv':
                continue

            conv = layer['functions']['conv']
            convState = conv.state_dict()

            # load batch norm parameters
            if 'batch_normalize' in layer['functions'].keys():
                batchNorm = layer['functions']['batch_normalize']
                batchNormState = batchNorm.state_dict()
                num = batchNorm.bias.numel()

                for attr, attrShape in zip(['bias', 'weight', 'running_mean', 'running_var'], \
                                       [batchNorm.bias,batchNorm.weight,batchNorm.running_mean,batchNorm.running_var]):
                    batchNormState[attr] = torch.from_numpy(weights[ptr:ptr+num]).view_as(attrShape).to(device)
                    ptr += num

                # save them
                batchNorm.load_state_dict(batchNormState)
            # or bias
            else:
                num = conv.bias.numel()
                convState['bias'] = torch.from_numpy(weights[ptr:ptr+num]).view_as(conv.bias).to(device)
                ptr += num

            # load weight
            num = conv.weight.numel()
            convState['weight'] = torch.from_numpy(weights[ptr:ptr+num]).view_as(conv.weight)
            ptr += num

            # save them
            conv.load_state_dict(convState)


def initWeight(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def createModelFromCfg(device, cfgPath, weightsPath=""):
    cfgFile = open(cfgPath, 'r')
    lines = cfgFile.readlines()

    layerProperties = []
    netPropertiesIndex = None
    for line in lines:
        # remove new line character
        line = line[0:-1]
        # skip comments and blank lines
        if line.startswith('#') or line == "":
            continue
        # new definition
        elif line.startswith('['):
            layerProperties.append({'type': line[1:-1]})
            if line == '[net]':
                netPropertiesIndex = len(layerProperties) - 1
        # layer properties
        else:
            key, value = line.split('=')
            layerProperties[-1][key.replace(' ', '')] = value.replace(' ', '')

    # copy net's properties
    netProps = dict(layerProperties[netPropertiesIndex])
    # then delete it from layers
    del(layerProperties[netPropertiesIndex])

    model = CustomModel(netProps, layerProperties).to(device)

    if weightsPath.endswith(".pth"):
        model.load_state_dict(torch.load(weightsPath, map_location=device))
    elif weightsPath != "":
        # init before loading weights since not all layers will loaded with the backbone model
        model.apply(initWeight)
        model.loadBackboneWeights(weightsPath, device)
    return model
