import argparse
import numpy as np
import os
import torch

from augmentations import DEFAULT_TRANSFORMS
from dataset import ListDataset, resize
from loss import calcLoss
from model import createModelFromCfg
import params


def main(previousStatePath):
    # get current device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: Cuda is not available, working on CPU!!")

    # create model from config with pretrained backbone
    models = ['yolo/darknet53.conv.74', 'yolo/yolov3.weights', 'save/yolov3_test.pth']
    useIdx = 0
    model = createModelFromCfg(device, 'yolo/yolov3.cfg', models[useIdx])

    # create dataloader
    dataset = ListDataset(params.cocoTrainPath, model.hyperparams['height'], DEFAULT_TRANSFORMS)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=model.hyperparams['learning_rate'], weight_decay=model.hyperparams['decay'], momentum=model.hyperparams['momentum'])

    epochsToTrain = 100

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochsToTrain*0.5), int(epochsToTrain*0.75), int(epochsToTrain*0.9)], gamma=0.1)

    imgIndex = 5
    img, label = dataset[imgIndex]
    print(label)
    img = resize(img, model.hyperparams['height'])
    img = img.unsqueeze(0).to(device)
    for epoch in range(epochsToTrain):
        # set model to training mode
        model.train()

        # pass img and label to current device
        label = label.to(device)

        # forward
        outputs = model(img)

        # cumpute loss
        loss = calcLoss(model, outputs, label)
        lossVal = float(loss)

        # backward feed loss
        loss.backward()

        # optimize net
        optimizer.step()
        # set grads to zero
        optimizer.zero_grad()

        scheduler.step()

        # change progress bar description
        print(f"Epoch {epoch}, loss: {lossVal:04.5f}", flush=True)

        torch.save(model.state_dict(), f"save/yolov3_test_{epoch}.pth")

    print(f"im={dataset.imgFiles[imgIndex]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--state', type=str, default=None, help="Path to saved state (model, optimizer, loss, epoch)")
    args = parser.parse_args()
    if not os.path.exists('save'): os.mkdir('save')
    if not os.path.exists('log'): os.mkdir('log')
    main(args.state)
