# Includes
from pathlib import Path

import h5py
import torch
from logger import VisdomLogger
from scipy import misc
import numpy as np
from utils import *
from models import TrainableModel, DataParallelModel
from modules.unet import UNetOld, UNet
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import ImageDataset
from modules.percep_nets import Dense1by1Net
from modules.depth_nets import UNetDepth

scenes_to_post_to_visdom = ['point_593_view_3_domain', 'point_648_view_2_domain', 'point_1262_view_2_domain',
                            'point_1141_view_1_domain', 'point_664_view_7_domain', 'point_1219_view_10_domain',
                            'point_902_view_9_domain', 'point_868_view_5_domain']
logger = VisdomLogger("train", env='baselinenormals_geonet')


def convert_to_taxonomy_normals(thing):
    return 255 - thing


def convert_normal_for_display(normal_image, should_invert=False):
    if type(normal_image) == torch.Tensor:
        normal_image = normal_image.cpu().numpy()

    if normal_image.shape[0] == 3:
        normal_image = normal_image.transpose([1, 2, 0])

    #     if any(val < 0 for val in normal_image.reshape([-1])):
    #         normal_image = (normal_image+1)/2

    #     normal_image = np.clip(normal_image, 0, 1)

    if should_invert:
        normal_image = 1.0 - normal_image

    return normal_image


# GeoNet on Taskonomy
def geonet_on_tasknonomy():
    rgb_loc = 'scaling/mount/data/taskonomy3/almena_rgb/rgb/{}_rgb.png'
    normals_loc = "scaling/mount/shared/baseline_outputs/geonet/taskonomy/almena/{}_normals_pred.npy"
    depths_loc = "scaling/mount/shared/baseline_outputs/geonet/taskonomy/almena/{}_depth_pred.npy"
    normals_gt_loc = "scaling/mount/data/taskonomy3/almena_normal/normal/{}_normal.png"

    taskonomy_rgb_images = []
    normal_images = []
    depth_images = []
    normal_gt_images = []

    # Upload RGB + normals + depth + GT normals all to visdom
    for scene_name in scenes_to_post_to_visdom:
        if type(scene_name) == str:
            rgb_image = misc.imread(rgb_loc.format(scene_name)).astype('uint8')
        else:
            rgb_image = misc.imread(scene_name).astype('uint8')

        taskonomy_rgb_images.append(rgb_image)

        normals_data = np.load(normals_loc.format(scene_name))
        normals_data = convert_to_taxonomy_normals(normals_data * 128 + 128).astype('uint8')
        normal_images.append(normals_data)

        depth_raw = np.load(depths_loc.format(scene_name))
        depth_raw = np.expand_dims(depth_raw[0], 2)
        depth_raw_2 = np.clip(depth_raw, 0, 10) / 10 * 255
        depth_images.append(depth_raw_2)

        normal_gt = misc.imread(normals_gt_loc.format(scene_name))
        normal_gt = (normal_gt).astype('uint8')
        normal_gt_images.append(normal_gt)

    logger.images(taskonomy_rgb_images, "taskonomy_rgb_inputs", resize=96 * 2)
    logger.images(normal_images, "taskonomy_normal_geonet_preds", resize=96 * 2)
    logger.images(depth_images, "taskonomy_depth_geonet_preds", resize=96 * 2)
    logger.images(normal_gt_images, "taskonomy_normal_gt", resize=96 * 2)


def geonet_on_nyuv2():
    nyu_rgb_images = []
    normal_images = []
    depth_images = []
    normal_gt_images = []

    nyu_norm_gt_file_name = '/home/ajaysohmshetty/geonet/data/norm_gt_l.mat'

    with h5py.File(nyu_norm_gt_file_name, 'r') as f:
        nyu_norm_gt = f['norm_gt_l']

        output_dir = Path("/home/ajaysohmshetty/scaling/mount/shared/baseline_outputs/geonet/nyu_v2/")
        for ii in range(8):
            rgb_output = output_dir / str(str(ii) + '_rgb_input.npy')
            normals_output = output_dir / str(str(ii) + '_normals_pred.npy')
            depth_output = output_dir / str(str(ii) + '_depth_pred.npy')

            rgb_raw = np.load(rgb_output)
            normals_raw = np.load(normals_output)
            normals_raw = convert_to_taxonomy_normals((normals_raw * 128 + 128)).astype('uint8')
            depth_raw = np.load(depth_output)
            depth_raw = np.expand_dims(depth_raw[0], 2)
            depth_raw = np.clip(depth_raw, 0, 10) / 10 * 255
            normals_gt_raw = convert_to_taxonomy_normals((nyu_norm_gt[ii].transpose(2, 1, 0) * 128 + 128)).astype(
                'uint8')

            nyu_rgb_images.append(rgb_raw)
            normal_images.append(normals_raw)
            depth_images.append(depth_raw)
            normal_gt_images.append(normals_gt_raw)

    logger.images(nyu_rgb_images, "nyu_rgb_inputs", resize=96 * 2)
    logger.images(normal_images, "nyu_normal_geonet_preds", resize=96 * 2)
    logger.images(depth_images, "nyu_depth_geonet_preds", resize=96 * 2)
    logger.images(normal_gt_images, "nyu_normal_gt", resize=96 * 2)


# Run our best models on (nyu, taskonomy dataset, and ood images)
def run_torch_model_on_test_data(model_name, images_to_predict):
    loaded_model = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/{model_name}")
    data = torch.Tensor([im.numpy() for im in images_to_predict])
    print(data.shape)
    predictions = loaded_model.predict([data])
    final_predictions = [t_prediction.cpu().numpy().transpose(1, 2, 0) for t_prediction in predictions]
    return [(np.clip(im, 0, 1) * 255).astype('uint8') for im in final_predictions]


def get_curv_depth_from_normals(images_normals):
    def run_model(model, data, saved_weights_file, scale=255):
        model = DataParallelModel.load(model, saved_weights_file)
        predictions = model.predict([data])
        predictions = [t_prediction.cpu().numpy().transpose(1, 2, 0) for t_prediction in predictions]
        return [(np.clip(im, 0, 1) * scale).astype('uint8') for im in predictions]

    data = [im.transpose(2, 0, 1) for im in images_normals]
    data = torch.Tensor(data)
    curv_preds = run_model(Dense1by1Net().cuda(), data, f"{MODELS_DIR}/normal2curvature_dense_1x1.pth")
    depth_preds = run_model(UNetDepth().cuda(), data, f"{MODELS_DIR}/normal2zdepth_unet_v4.pth", scale=512)

    return curv_preds, depth_preds


def our_approach_on_taskonomy_and_nyuv2():
    models_to_generate_from = ["mixing_percepcurv_norm.pth", "unet_percepstep_0.1.pth"]

    ood_dataset = ImageDataset(data_dir='data/ood_images')
    taskonomy_dataset = ImageDataset(data_dir='./mount/data/taskonomy3/almena_rgb/rgb/')
    taskonomy_dataset.files = [f"./mount/data/taskonomy3/almena_rgb/rgb/{scene_name}_rgb.png" for scene_name in
                               scenes_to_post_to_visdom]
    geonet_dataset = ImageDataset(data_dir="/home/ajaysohmshetty/scaling/mount/shared/baseline_outputs/geonet/nyu_v2/")
    geonet_dataset.files = [file_loc for file_loc in geonet_dataset.files if
                            int(file_loc.split('/')[-1].split('_')[0]) < 8]

    for model_to_generate_from in models_to_generate_from:
        print(model_to_generate_from)
        taskonomy_preds = run_torch_model_on_test_data(model_to_generate_from, taskonomy_dataset)
        nyu_preds = run_torch_model_on_test_data(model_to_generate_from, geonet_dataset)
        nyu_curv_preds, nyu_depth_preds = get_curv_depth_from_normals(nyu_preds)
        ood_preds = run_torch_model_on_test_data(model_to_generate_from, ood_dataset)

        logger.images(taskonomy_preds, "taskonomy_normal_{}_preds".format(model_to_generate_from), resize=96 * 2)
        logger.images(nyu_preds, "nyu_normal_{}_preds".format(model_to_generate_from), resize=96 * 2)
        logger.images(nyu_depth_preds, "nyu_depth_from_{}_normals_preds".format(model_to_generate_from), resize=96 * 2)
        logger.images(nyu_curv_preds, "nyu_curv_from_{}_normals_preds".format(model_to_generate_from), resize=96 * 2)
        logger.images(ood_preds, "ood_normal_{}_preds".format(model_to_generate_from), resize=96 * 2)


def upsampling_experiment():
    geonet_dataset = ImageDataset(data_dir="/home/ajaysohmshetty/scaling/mount/shared/baseline_outputs/geonet/nyu_v2/")
    geonet_dataset.files = [file_loc for file_loc in geonet_dataset.files if
                            int(file_loc.split('/')[-1].split('_')[0]) < 8]

    # Upsampling on our augment2d
    # upsample_sizes = [(256+64*ii, 256+64*ii) for ii in range(4)]
    upsample_sizes = [(448 - 64 * ii, 640 - 64 * ii) for ii in range(4)]
    print(upsample_sizes)
    model_to_generate_from = "augmented_base2.pth"

    for upsample_size in upsample_sizes:
        geonet_dataset_upsampled = ImageDataset(
            data_dir="/home/ajaysohmshetty/scaling/mount/shared/baseline_outputs/geonet/nyu_v2/",
            resize=upsample_size)
        geonet_dataset_upsampled.files = [file_loc for file_loc in geonet_dataset.files if
                                          int(file_loc.split('/')[-1].split('_')[0]) < 8]
        nyu_preds = run_torch_model_on_test_data(model_to_generate_from, geonet_dataset_upsampled)
        logger.images(nyu_preds, "nyu_normal_upsampled_{}_{}_preds".format(str(upsample_size), model_to_generate_from),
                      resize=96 * 2)
