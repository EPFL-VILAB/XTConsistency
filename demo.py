import torch
from torchvision import transforms
from modules.unet import UNet, UNetReshade
import PIL
from PIL import Image


####### TODO #######
# 1. rename file paths
# 2. remove state dict key rename after renaming those in .pth files
# 3. merge baseline and consistency output code

# open image
root_dir = '/scratch-data/shared/'
img = Image.open(root_dir + 'ood_standard_set/abbasi-hotel-safavid-suite.png')
trans_totensor = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
img_tensor = trans_totensor(img)[:3].unsqueeze(0)
trans_topil = transforms.ToPILImage()

# define target task
target_tasks = ['normal','depth','reshading']
models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]


for target_task, model in zip(target_tasks,models):

    # compute baseline output
    path = root_dir + 'results_CH_baseline_unet_'+target_task+'target_nosqerror_dataaug_contlr3e-5_1/model.pth'
    model_state_dict = torch.load(path)
    model_state_dict = {k[22:]: v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    baseline_output = model(img_tensor)

    # compute consistency output
    path = root_dir + 'results_CH_lbp_all_'+target_task+'target_gradnorm_unnormalizedmse_imagenet_nosqerror_nosqinitauglr_dataaug_1/graph.pth'
    if target_task == 'depth': target_task = 'depth_zbuffer'
    model_state_dict = torch.load(path)["('rgb', '"+target_task+"')"]
    model_state_dict = {k[28:]: v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    consistency_output = model(img_tensor)

    # save image
    out_stacked = Image.new('RGB', (256*3, 256))
    x_offset = 0
    for im in [img_tensor,baseline_output,consistency_output]:
      out_stacked.paste(trans_topil(im[0]), (x_offset,0))
      x_offset += im.size(2)

    out_stacked.save(target_task+'_out.jpg')
