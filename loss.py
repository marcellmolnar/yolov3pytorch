import numpy as np
import torch
import torch.nn as nn

weightBox = 0.05
weightClass = 3.0
weightObjectness = 3.0

eps = 1e-10


def isOut(v):
    return torch.isnan(v) or torch.isinf(v)


def createTargets(yoloLayers, yoloLayerShapes, labels, device):
    targets = []
    targetPositions = [[] for _ in range(len(yoloLayers))]
    anchorSizes = torch.zeros((len(yoloLayers),3))
    imageSize = yoloLayers[0].fullImageSize
    for i, yoloLayer in enumerate(yoloLayers):
        targets.append(torch.zeros(yoloLayerShapes[i],device=device))
        for anchorIdx, a in enumerate(yoloLayer.anchors):
            anchorSizes[i,anchorIdx] = a[0]*a[1]

    anchorImageSize = 416
    for label in labels:
        batchIdx, classId, posX, posY, width, height = label
        boxSize = float(width * height * anchorImageSize * anchorImageSize)
        ratios = np.minimum(boxSize / anchorSizes, anchorSizes / boxSize)
        bestAnchorIdx = int(np.argmax(ratios))
        i = int(bestAnchorIdx / 3)
        bestAnchorIdx = bestAnchorIdx % 3
        yoloLayer = yoloLayers[i]
        #print(f"box size: {boxSize}\nanch sizes:\n{anchorSizes}\nratios:\n{ratios}\nlayer: {i}, anch: {bestAnchorIdx}, layer shape: {yoloLayerShapes[i][2]}x{yoloLayerShapes[i][3]}")
        if True:
        #for i, yoloLayer in enumerate(yoloLayers):
            #ratios = np.minimum(boxSize / anchorSizes[i,:], anchorSizes[i,:] / boxSize)
            #bestAnchorIdx = np.argmax(ratios) # ratios > 0.25
            #if ratios[bestAnchorIdx] < 0.25:
            #    continue

            anchorPosX = int(torch.floor(posX * yoloLayerShapes[i][3]))
            posXWithinBox = posX * yoloLayerShapes[i][3] - anchorPosX
            #print(f"shape {yoloLayerShapes[i][3]}, posX {posX}, anchorPosX {anchorPosX}, posXWithinBox {posXWithinBox}")
            anchorPosY = int(torch.floor(posY * yoloLayerShapes[i][2]))
            posYWithinBox = posY * yoloLayerShapes[i][2] - anchorPosY

            #if posXWithinBox<0.001 or posYWithinBox<0.001 or posXWithinBox>0.999 or posYWithinBox>0.999:
            #    continue

            batchIdx = int(batchIdx)

            if targets[i][batchIdx,bestAnchorIdx,anchorPosY,anchorPosX,4] == 1.0:
                continue

            #valx = posXWithinBox
            #valy = posYWithinBox
            valx = torch.log(eps + posXWithinBox / (1 - posXWithinBox + eps))
            valy = torch.log(eps + posYWithinBox / (1 - posYWithinBox + eps))
            valw = torch.log(width * anchorImageSize / yoloLayer.anchors[bestAnchorIdx,0].to(device)).to(device)
            valh = torch.log(height * anchorImageSize / yoloLayer.anchors[bestAnchorIdx,1].to(device)).to(device)

            #print(f"batch: {batchIdx}, anc x: {anchorPosX}, anc y: {anchorPosY}, x: {valx}, y: {valy}, w: {valw}, h: {valh}")
            if isOut(valx) or isOut(valy) or isOut(valw) or isOut(valh):
                print(f"{valx:+04.5f} {valy:+04.5f} {valw:+04.5f} {valy:+04.5f} ({posX:04.5f} {posY:04.5f} {width:04.5f} {height:04.5f}) c: {classId}")
                continue

            targets[i][batchIdx,bestAnchorIdx,anchorPosY,anchorPosX,0] = valx
            targets[i][batchIdx,bestAnchorIdx,anchorPosY,anchorPosX,1] = valy
            targets[i][batchIdx,bestAnchorIdx,anchorPosY,anchorPosX,2] = valw
            targets[i][batchIdx,bestAnchorIdx,anchorPosY,anchorPosX,3] = valh

            targets[i][batchIdx,bestAnchorIdx,anchorPosY,anchorPosX,4] = 1.0  # objectness
            targets[i][batchIdx,bestAnchorIdx,anchorPosY,anchorPosX,5+int(classId)] = 1.0  # class

            targetPositions[i].append((batchIdx,bestAnchorIdx,anchorPosY,anchorPosX))
            #break

    return targets, targetPositions


def calcLoss(model, outputs, labels):
    device = outputs[0].device
    yoloLayerShapes = [outputs[i].shape for i in range(len(outputs))]

    targets, targetPositions = createTargets(model.yoloLayers, yoloLayerShapes, labels, device)

    lossBox = torch.zeros(1, device=device)
    lossObjectness = torch.zeros(1, device=device)
    lossClass = torch.zeros(1, device=device)

    BCE_class = nn.BCELoss()
    BCE_objectness = nn.BCELoss()
    MSE_box = nn.MSELoss()

    for i, _ in enumerate(model.yoloLayers):
        if targetPositions[i]:
            # list of indices for each dimension
            b,a,y,x = [torch.tensor(targetPositions[i])[:,j] for j in range(4)]

            # calc box and class prediction loss for all target positions
            vbox = MSE_box(outputs[i][b,a,y,x,0:4], targets[i][b,a,y,x,0:4])
            lossBox += vbox

            vcls = BCE_class(outputs[i][b,a,y,x,5:].flatten(), targets[i][b,a,y,x,5:].flatten())
            lossClass += vcls

        # calc objectness loss for all positions
        vobj = BCE_objectness(outputs[i][..., 4].flatten(), targets[i][..., 4].flatten())
        lossObjectness += vobj

    return weightBox * lossBox + weightObjectness * lossObjectness + weightClass * lossClass
