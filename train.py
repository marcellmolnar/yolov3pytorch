import argparse
import numpy as np
import os
import random
import torch
import tqdm

from dataset import createDataLoader
from loss import calcLoss
from model import createModelFromCfg
import params
from test import evaluateModel

evalInterval = 1

def main(previousStatePath, epochsToTrain):
    # seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # get current device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: Cuda is not available, working on CPU!!")

    # create model from config with pretrained backbone
    model = createModelFromCfg(device, 'yolo/yolov3.cfg', 'yolo/darknet53.conv.74')

    # create dataloader
    dataloader = createDataLoader(params.cocoTrainPath, model.hyperparams['batch'], model.hyperparams['height'])

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=model.hyperparams['learning_rate'], weight_decay=model.hyperparams['decay'], momentum=model.hyperparams['momentum'])

    epochsDone = 0

    if previousStatePath is not None:
        checkpoint = torch.load(previousStatePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochsDone = checkpoint['epoch'] + 1

    # set model to training mode
    model.train()

    for epoch in range(epochsDone, epochsDone + epochsToTrain):

        # enumerate through all training examples
        progressBar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader))
        for images, labels in progressBar:

            # set grads to zero
            optimizer.zero_grad()

            # pass images and labels to current device
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)

            # cumpute loss
            loss = calcLoss(model, outputs, labels)
            lossVal = float(loss)

            # backward feed loss
            loss.backward()

            # optimize net
            optimizer.step()

            # change progress bar description
            progressBar.set_description(f"Epoch {epoch}, loss: {lossVal:04.5f}")

        torch.save(model.state_dict(), f"save_v2/yolov3_epoch_{epoch}.pth")

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f"save_v2/state_save_epoch_{epoch}")

        # evaluate model and write log
        if epoch % evalInterval == 0:
            evaluateModel(model,device,epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--state', type=str, default=None, help="Path to saved state (model, optimizer, loss, epoch)")
    parser.add_argument('-e', '--epochs', type=int, required=True, help="Number of epochs to train")
    args = parser.parse_args()
    if not os.path.exists('save'): os.mkdir('save')
    if not os.path.exists('save_v2'): os.mkdir('save_v2')
    if not os.path.exists('log'): os.mkdir('log')
    main(args.state, args.epochs)
