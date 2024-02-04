import argparse
import numpy as np
import os

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image, ImageDraw, ImageFont

from dataset import resize
from model import createModelFromCfg
import params
from test import filterOutputs, load_classes


def loadImage(img_path, device):
    img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    img = transforms.ToTensor()(img).to(device)
    return img

def getModel(device, model_path, weights_path):
    model = createModelFromCfg(device, model_path, weights_path)
    model.eval()
    return model

def getOutput(model, img):
    img = resize(img, model.hyperparams['height'])
    outputs = model(torch.stack([img]))
    fl = tuple(output.reshape(1, -1, 85) for output in outputs)
    stackedOutputs = torch.concat(fl, 1)
    outputs = filterOutputs(stackedOutputs)
    return outputs

plt.rcParams["savefig.bbox"] = 'tight'
def show(imgs, labels, output_path, only_save_image):
    plt.close()
    if not isinstance(imgs, list):
        imgs = [imgs]
    
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        draw = ImageDraw.Draw(img)

        for l in labels[i]:
            font = ImageFont.truetype("arial.ttf" if os.name=="nt" else "FreeMono.ttf", l['size'])
            draw.text((l['x'], l['y']), l['str'], l['color'], font=font)

        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if only_save_image:
        plt.savefig(output_path)
    else:
        plt.show()

colors_org = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#C0C0C0",
          "#808080", "#800000", "#808000", "#008000", "#800080", "#008080", "#000080"]
colors=[    
"#FF0000",
"#00FF00",
"#0000FF",
"#FFFF00",
"#00FFFF",
"#FF00FF",
"#000000",
"#800000",
"#008000",
"#808000",
"#000080",
"#800080",
"#008080",
"#C0C0C0",
"#808080",
"#FFFFFF"]
def getColor(idx):
    return colors[idx % len(colors)]


def visualizeResult(img_path, output, output_path, only_save_image = False):
    from torchvision.io import read_image
    img = read_image(img_path)
    img_h, img_w = img.shape[1], img.shape[2]
    class_names = load_classes(params.cocoClassNames)  # List of class names

    labels = []
    """
    objectsToShow = 15
    # threshold that will allow 'objectsToShow' number of objects
    #scores = -np.sort(-output[...,4].cpu().detach().numpy())
    #th = max(float(scores[0,objectsToShow]), scores[0,0] * 0.5)
    th=0.1
    if not only_save_image:
        print(f"     class name     prob            posx     posy           width  height")
        
    for o in output[0]:
        if o[4] > th:
            xc, yc = o[0] * img_w, o[1] * img_h
            w,h = o[2] * img_w, o[3] * img_h
            box = torch.tensor([[xc-w/2, yc-h/2, xc+w/2, yc+h/2]])

            classIdx = torch.argmax(o[5:])
            className = class_names[classIdx]
            if not only_save_image:
                print(f"{className:>15s}: {o[4]:1.5f}    pos:  {xc:6.2f}   {yc:6.2f}    size:  {h:6.2f}  {w:6.2f}   class: {o[5+classIdx]:5.3f} min: {torch.min(o[5:]):5.3f} max: {torch.max(o[5:]):5.3f}")
            img = draw_bounding_boxes(img, box, colors=getColor(classIdx), width=5)
            labels.append({'x':xc-w/2, 'y':yc-h/2, 'str': className, 'color': getColor(classIdx), 'size': int(img_h/20)})
    """
    
    with open(os.path.join(output_path, os.path.basename(os.path.splitext(os.path.basename(img_path))[0])+".txt"), "w") as f:
        for o in output[0]:
            xc, yc = o[0] * img_w, o[1] * img_h
            w,h = o[2] * img_w, o[3] * img_h
            box = torch.tensor([[xc-w/2, yc-h/2, xc+w/2, yc+h/2]])

            # class already calculated
            classIdx = int(o[5])
            className = class_names[classIdx]
            img = draw_bounding_boxes(img, box, colors=getColor(classIdx), width=5)
            conf=o[4]*100
            labels.append({'x':xc-w/2+4, 'y':yc-h/2, 'str': f"{className}: {conf:02.2f} %", 'color': getColor(classIdx), 'size': int(img_h/15)})
            
            f.write(f"{className} x,y: {xc:06.2f}, {yc:06.2f}, w,h: {w:06.2f}, {h:06.2f}, conf: {conf:02.2f} %\n")

    show([img], [labels], os.path.join(output_path, os.path.basename(img_path)), only_save_image)

def runOnImages(device, model_path, weights_path, images_path, output_path):
    model = getModel(device, model_path, weights_path)
    imList = os.listdir(images_path)
    totalImages = len(imList)
    for i, filename in enumerate(imList):
        if i % 100 == 0: # int(totalImages*0.05)
            print(f"{i}/{totalImages} done")
        #if i > 10:
        #    break
        fName = os.path.join(images_path, filename)
        if not os.path.isfile(fName):
            continue
        img = loadImage(fName, device)
        output = getOutput(model, img)
        visualizeResult(fName, output, output_path, only_save_image=True)

def run():
    parser = argparse.ArgumentParser(description="Run model on image and visualize result.")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-i", "--image", type=str, default="", help="Path to the image to run the model on.")
    parser.add_argument("-d", "--dir", type=str, default="", help="Path to the images to run the model on.")
    parser.add_argument("-o", "--output", type=str, default="", help="Path where to save the inferenced images.")
    args = parser.parse_args()
    
    # get current device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: Cuda is not available, working on CPU!!")

    if args.image != "":
        img = loadImage(args.image, device)
        model = getModel(device, params.yolov3CfgFile, args.weights)
        output = getOutput(model, img)
        visualizeResult(args.image, output, "", False)
    elif args.dir != "":
        if args.output == "":
            print("When -d argument is used, -o (--output) also shall be provided.")
            return
        if not os.path.exists(args.output): os.mkdir(args.output)
        runOnImages(device, params.yolov3CfgFile, args.weights, args.dir, args.output)
    else:
        print("At least -i (--image) or -d (--dir) argument shall be provided")

if __name__ == "__main__":
    run()
