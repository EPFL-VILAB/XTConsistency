import torch
from torchvision import transforms

from modules.unet import UNet, UNetReshade

import PIL
from PIL import Image

import argparse
import os.path
from pathlib import Path
import glob

import pdb


####### TODO #######
# 1. rename file paths
# 2. remove state dict key rename after renaming those in .pth files
# 3. merge baseline and consistency output code


parser = argparse.ArgumentParser(description='Visualize output for a single Task')

parser.add_argument('--task', dest='task', help="normal, depth or reshading")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

root_dir = '/scratch-data/shared/'
trans_totensor = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
trans_topil = transforms.ToPILImage()


# get target task and model
target_tasks = ['normal','depth','reshading']
task_index = target_tasks.index(args.task)
models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]
target_task = args.task
model = models[task_index]


def save_outputs(img_path, output_file_name):

    img = Image.open(img_path)
    img_tensor = trans_totensor(img)[:3].unsqueeze(0)

    # compute baseline output
    path = root_dir + 'results_CH_baseline_unet_'+args.task+'target_nosqerror_dataaug_contlr3e-5_1/model.pth'
    model_state_dict = torch.load(path)
    model_state_dict = {k[22:]: v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    baseline_output = model(img_tensor)
    trans_topil(baseline_output[0]).save(args.output_path+'/'+output_file_name+'_'+args.task+'_baseline'+'.jpg')

    # compute consistency output
    path = root_dir + 'results_CH_lbp_all_'+args.task+'target_gradnorm_unnormalizedmse_imagenet_nosqerror_nosqinitauglr_dataaug_1/graph.pth'
    alt_name = args.task+'_zbuffer' if args.task == 'depth' else args.task
    model_state_dict = torch.load(path)["('rgb', '"+alt_name+"')"]
    model_state_dict = {k[28:]: v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    consistency_output = model(img_tensor)
    trans_topil(consistency_output[0]).save(args.output_path+'/'+output_file_name+'_'+args.task+'_consistency'+'.jpg')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*'):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()
