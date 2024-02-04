import argparse
import numpy as np
import os
import torch
import torchvision
import tqdm

from dataset import createDataLoader
from model import createModelFromCfg
import params

iouThres = 0.5
confThres = 0.01
nmsThres = 0.3

def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().splitlines()
    return names

def calcIoU(xy1, wh1, xy2, wh2):
    intersectionX = torch.clamp(torch.min(xy1[:,0]+wh1[:,0]/2,xy2[:,0]+wh2[:,0]/2) - torch.max(xy1[:,0]-wh1[:,0]/2,xy2[:,0]-wh2[:,0]/2), 0)
    intersectionY = torch.clamp(torch.min(xy1[:,1]+wh1[:,1]/2,xy2[:,1]+wh2[:,1]/2) - torch.max(xy1[:,1]-wh1[:,1]/2,xy2[:,1]-wh2[:,1]/2), 0)
    intersection = intersectionX*intersectionY
    union = wh1[:,0]*wh1[:,1] + wh2[:,0]*wh2[:,1]
    return intersection / (union - intersection + 1e-10)

class Metrics:
    """ Metrics per class """
    numberGTs = np.zeros(80)
    dtype = [('confidence', float),('isTP', bool)]
    detections = np.repeat(np.array([],dtype=dtype), 80)

    def reset():
        Metrics.numberGTs = np.zeros(80)
        Metrics.detections = [np.array([],dtype=Metrics.dtype)] * 80

    def calculateMAP():
        # magic: for each class: we order the detections by confidence (increasing, that's why the flip)
        #  and then we get only the isTP values and finally we create a cumsum
        cummulativeTPs = [np.flip(np.sort(dets, order='confidence'))['isTP'].cumsum() if len(dets) > 0 else np.zeros(1) for dets in Metrics.detections]

        # precision is the #TP / #detections
        precisions = [dets / np.arange(1,1+len(dets)) if len(dets) > 0 else np.zeros(len(dets)) for dets in cummulativeTPs]

        # recall is the #TP / #GT
        recalls = [dets / GTs if len(dets) > 0 and GTs > 0 else np.zeros(len(dets)) for dets,GTs in zip(cummulativeTPs,Metrics.numberGTs)]

        mAP = 0
        APs = np.zeros(80)

        # to calculate mAP, first calculate AP for each class individually
        for classIndex in range(80):
            classPrecision = precisions[classIndex]
            classRecall = recalls[classIndex]

            # to compensate noise in the precision-recall curve, use a 11 point interpolation (with 10 range)
            envelope = np.zeros(10)
            for i in range(len(classPrecision)):
                rangeIndex = int(np.floor(classRecall[i]*10)) % 10
                maxToRight = np.max(classPrecision[i:])
                envelope[rangeIndex] = max(envelope[rangeIndex], maxToRight)

            # the area below the precision-recall curve is the average of the envelope values
            AP = envelope.sum() / 10
            APs[classIndex] = AP

            # mAP increment
            mAP += AP / 80

        return mAP, APs

def collectMetrics(outputs, groundTruths):
    # count ground truth objects
    for gt in groundTruths:
        Metrics.numberGTs[int(gt[1])] += 1

    # check detections and add them
    for batchIdx, imageDetections in outputs.items():
        imageGTs = groundTruths[groundTruths[:,0]==batchIdx]
        found = np.repeat(False,len(imageGTs))
        for output in imageDetections:
            predClass = int(output[5])
            isTP = False
            detxy = output[0:2].reshape(1,2)
            dethw =  output[2:4].reshape(1,2)
            for i,gt in enumerate(imageGTs):
                gtxy = gt[2:4].reshape(1,2)
                gthw = gt[4:6].reshape(1,2)
                if int(gt[1]) == predClass and not found[i] and calcIoU(detxy, dethw, gtxy, gthw) > iouThres:
                    isTP = True
                    found[i] = True
            newDetection = np.array([(output[4],isTP)], dtype=Metrics.dtype)
            Metrics.detections[predClass] = np.append(Metrics.detections[predClass], newDetection)

def filterOutputs(outputs):
    filteredOutputs = {}
    for batchIdx, imageOutputs in enumerate(outputs):
        # discard outputs with low confidence
        imageOutputs = imageOutputs[imageOutputs[:, 4] > confThres]

        # set the sixth value for the predicted class and remove the class predictions
        val = torch.argmax(imageOutputs[:, 5:], 1)
        imageOutputs[:, 5] = val
        imageOutputs = imageOutputs[:, :6]

        mask = torch.zeros(len(imageOutputs), dtype=torch.bool)
        # non-max suppression in each class
        for classIdx in torch.unique(imageOutputs[:, 5]):
            classFilter = (imageOutputs[:, 5] == classIdx).nonzero()

            # transform xywh into xy1xy2
            xywh = imageOutputs[classFilter, :4].squeeze(1)
            xy1xy2 = xywh.clone()
            xy1xy2[:,0:2] = xywh[:,0:2] - xywh[:,2:4] / 2
            xy1xy2[:,2:4] = xywh[:,0:2] + xywh[:,2:4] / 2

            toKeep = torchvision.ops.nms(xy1xy2, imageOutputs[classFilter, 4].squeeze(1), nmsThres)
            mask[classFilter[toKeep]] = True
        filteredOutputs[batchIdx] = imageOutputs[mask, :].detach().cpu()
    return filteredOutputs

def evaluateModel(model, device, epoch=None):
    model.eval()

    batchSize = model.hyperparams['batch']

    dataLoader = createDataLoader(params.cocoValidPath, batchSize, model.hyperparams['height'], validation=True)

    Metrics.reset()
    for images, groundTruths in tqdm.tqdm(dataLoader, desc="Validating model"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            fl = tuple(output.reshape(len(output), -1, 85) for output in outputs)
            stackedOutputs = torch.concat(fl, 1)
            outputs = filterOutputs(stackedOutputs)
        collectMetrics(outputs, groundTruths)
    
    mAP, APs = Metrics.calculateMAP()

    class_names = load_classes(params.cocoClassNames)
    aps = [f"{class_names[i]}: {a:0.3f}\n" for i,a in enumerate(APs)]
    #aps = [f"{a:0.3f}\n" for i,a in enumerate(APs)]
    print(f"mAP: {mAP}\n {aps}")

    if epoch is not None:
        with open(f"log/log_epoch_{epoch}.txt", "w") as f:
            f.write(f"mAP: {mAP}\n")
            for i in range(80):
                dets = Metrics.detections[i]
                lenOrg = len(dets)
                tps = len(dets[dets['isTP']])
                f.write(f"{i} AP: {APs[i]} gts: {Metrics.numberGTs[i]}, dets: {lenOrg}, tps: {tps}, fps: {lenOrg-tps}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing yolov3")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to weights")
    args = parser.parse_args()

    # get current device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: Cuda is not available, working on CPU!!")

    if not os.path.exists('log'): os.mkdir('log')

    model = createModelFromCfg(device, 'yolo/yolov3.cfg', args.weights)
    evaluateModel(model, device, 12345)
