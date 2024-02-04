# Yolov3 Academy

## Setup

Create a venv and install dependencies from ```requirements.txt```

Use ```download_files.sh``` to download the COCO dataset and the pretrained darknet backbone model.

## Running the script

You can use the ```train.py``` and ```test.py``` scripts to train and test the models. Each scripts have the ```-w``` or ```--weights``` parameter to specify the pretrained weights, which can be a torch state dictionary (.pth file) or a darknet weight file (file name ending with a number).

The ```train.py``` have an additional parameter ```-l``` or ```--logdir``` which can specify the directory where the logs will be put.

The trained model can be tested visually with the ```run_on_image.py``` script. Beside the weight parameter, it requires a ```-i``` or ```--image``` parameter that specifies the test image location.
