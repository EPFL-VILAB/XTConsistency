import torch
from torchvision import transforms

from modules.unet import UNet, UNetReshade

import PIL
from PIL import Image

import argparse
import os.path
from pathlib import Path
import glob
import sys

import pdb



parser = argparse.ArgumentParser(description='Visualize output for a single Task')

parser.add_argument('--task', dest='task', help="normal, depth or reshading")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

root_dir = './models/'
trans_totensor = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
trans_topil = transforms.ToPILImage()


# get target task and model
target_tasks = ['normal','depth','reshading']
try:
    task_index = target_tasks.index(args.task)
except:
    print("task should be one of the following: normal, depth, reshading")
    sys.exit()
models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]
model = models[task_index]


def save_outputs(img_path, output_file_name):

    img = Image.open(img_path)
    img_tensor = trans_totensor(img)[:3].unsqueeze(0)

    # compute baseline and consistency output
    for type in ['baseline','consistency']:
        path = root_dir + 'rgb2'+args.task+'_'+type+'.pth'
        model_state_dict = torch.load(path)
        model.load_state_dict(model_state_dict)
        baseline_output = model(img_tensor)
        trans_topil(baseline_output[0]).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+type+'.png')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*'):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()
