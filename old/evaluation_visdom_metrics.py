# Includes
from functools import lru_cache
from itertools import chain
from pathlib import Path

import fire
import imageio
from PIL import Image
from evaluation_visdom_visualizations import convert_normal_for_display
from logger import VisdomLogger
from models import DataParallelModel
from modules.unet import UNetOld, UNet
from skimage.transform import resize
from torch.utils import data
from torchvision import transforms
from utils import *
import scipy
import h5py

logger = VisdomLogger("train", env='baselinenormals_geonet')


def norm(a):
    return np.sqrt((a * a).sum(axis=1))


def my_cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = np.sum(x1 * x2, dim)
    w1 = norm(x1)
    w2 = norm(x2)
    return (w12 / (w1 * w2).clip(min=eps)).clip(min=-1.0, max=1.0)


def compute_ang_distances(pred, target):
    output = my_cosine_similarity(pred, target)
    return np.arccos(output) * 180 / math.pi


def get_metrics(original_pred, original_target, masks):
    original_pred = original_pred.astype('float64')
    original_target = original_target.astype('float64')

    print(original_target.shape, original_pred.shape)
    num_examples, width, height, num_channels = original_pred.shape

    pred = original_pred.reshape([-1, num_channels])
    target = original_target.reshape([-1, num_channels])

    ang_errors_per_pixel_unraveled = compute_ang_distances(pred, target)
    ang_errors_per_pixel = ang_errors_per_pixel_unraveled.reshape(num_examples, width, height)

    assert ang_errors_per_pixel.shape == masks.shape
    num_valid_pixels = np.sum(masks)
    num_invalid_pixels = np.sum(1 - masks)

    ang_errors_per_pixel_masked = ang_errors_per_pixel * masks

    ang_error_mean = np.sum(ang_errors_per_pixel_masked) / num_valid_pixels
    ang_error_without_masking = np.mean(ang_errors_per_pixel)
    ang_error_median = np.mean(np.median(np.ma.masked_equal(ang_errors_per_pixel_masked, 0), axis=1))

    normed_pred = pred / (norm(pred)[:, None] + 2e-1)
    normed_target = target / (norm(target)[:, None] + 2e-1)
    masks_expanded = np.expand_dims(masks, 3).reshape([-1])
    mse = (normed_pred - normed_target) * masks_expanded[:, None]
    mse = np.mean(mse ** 2)
    rmse = np.sqrt(mse) * 255

    threshold_1125 = (np.sum(ang_errors_per_pixel_masked <= 11.25) - num_invalid_pixels) / num_valid_pixels
    threshold_225 = (np.sum(ang_errors_per_pixel_masked <= 22.5) - num_invalid_pixels) / num_valid_pixels
    threshold_30 = (np.sum(ang_errors_per_pixel_masked <= 30) - num_invalid_pixels) / num_valid_pixels
    return {
        "ang_error_without_masking": ang_error_without_masking,
        "ang_error_mean": ang_error_mean,
        "ang_error_median": ang_error_median,
        # "rmse": rmse,
        'percentage_within_11.25_degrees': threshold_1125,
        'percentage_within_22.5_degrees': threshold_225,
        'percentage_within_30_degrees': threshold_30,
    }


class TaskonomyTestDataset(data.Dataset):
    def __init__(self, building, resize_frame=(256, 256)):
        def crop(x):
            return transforms.CenterCrop(min(x.size[0], x.size[1]))(x)

        self.transforms = transforms.Compose([crop, transforms.Resize(resize_frame), transforms.ToTensor()])
        self.image_paths = []
        self.building = building
        self.resize_frame = resize_frame

        for image_path in glob.glob("/home/ajaysohmshetty/scaling/mount/data/taskonomy3/{}_rgb/rgb/*".format(building)):
            image_name = str(image_path).split('/')[-1].split('.png')[0].split('_rgb')[0]
            normals_gt_loc = "/home/ajaysohmshetty/scaling/mount/data/taskonomy3/{}_normal/normal/{}_normal.png".format(
                building, image_name)

            if not Path(normals_gt_loc.format(image_name)).exists():
                continue

            if Path(image_path).stat().st_size == 0:
                continue

            self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def _open_png_and_transform(self, file):
        image = Image.open(file)
        image = self.transforms(image).float()[0:3, :, :]
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)

        return image

    @staticmethod
    def build_mask(target, val=0.502, tol=1e-3):
        mask1 = (target[0, :, :] >= val - tol) & (target[0, :, :] <= val + tol)
        mask2 = (target[1, :, :] >= val - tol) & (target[1, :, :] <= val + tol)
        mask3 = (target[2, :, :] >= val - tol) & (target[2, :, :] <= val + tol)
        mask = (mask1 & mask2 & mask3)
        mask = (~mask)
        return mask

    @staticmethod
    def fetch_mask_for_normal(normal_path, resize_frame):
        mask_image = imageio.imread(normal_path)  # TaskonomyTestDataset.build_mask()
        valid_pixels = TaskonomyTestDataset.build_mask(np.asarray(mask_image.transpose(2, 0, 1)) / 255).astype(float)
        valid_pixels = torch.Tensor(np.round(resize(valid_pixels, resize_frame)))
        return valid_pixels

    @lru_cache(maxsize=None)
    def __getitem__(self, index):  # return gt_image, normal_image, image_name
        image_path = self.image_paths[index]
        image_name = str(image_path).split('/')[-1].split('.png')[0].split('_rgb')[0]
        normal_path = "/home/ajaysohmshetty/scaling/mount/data/taskonomy3/{}_normal/normal/{}_normal.png".format(
            self.building, image_name)
        rgb_image = self._open_png_and_transform(image_path)
        normal_image = self._open_png_and_transform(normal_path)

        valid_pixels = TaskonomyTestDataset.fetch_mask_for_normal(normal_path, self.resize_frame)

        return rgb_image, normal_image, valid_pixels, image_name


class TaskonomyPredictionGenerator:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_predictions_from(model_name, building_name, dry_run=False, custom_data_loader=None):
        if custom_data_loader:
            data_loader = custom_data_loader
        else:
            taskonomy_test_dataset = TaskonomyTestDataset('almena')
            data_loader = data.DataLoader(taskonomy_test_dataset, batch_size=64)

        model_to_load = our_best_models_to_model[model_name]
        loaded_model = DataParallelModel.load(model_to_load.cuda(),
                                              f"/home/ajaysohmshetty/scaling/{MODELS_DIR}/{model_name}")

        batch_predictions = []
        batch_targets = []
        batch_valid_pixels = []
        batch_image_names = []
        for ii, (rgb_images, normal_images, valid_pixels, image_names) in enumerate(data_loader):
            if ii % 10 == 0:
                print(ii)
            if dry_run and ii > 2:
                break

            batch_predictions.append(loaded_model.predict([rgb_images]))
            batch_targets.append(normal_images)
            batch_valid_pixels.append(valid_pixels)
            batch_image_names.append(image_names)

        all_predictions = torch.cat(batch_predictions, dim=0).cpu().numpy().transpose(0, 2, 3, 1)
        all_targets = torch.cat(batch_targets, dim=0).cpu().numpy().transpose(0, 2, 3, 1)
        all_valid_pixels = torch.cat(batch_valid_pixels, dim=0).cpu().numpy()
        all_image_names = list(chain(*batch_image_names))

        return all_predictions, all_targets, all_valid_pixels, all_image_names

    @staticmethod
    @lru_cache(maxsize=None)
    def yeild_geonet_predictions_and_gts(building_name):
        return list(TaskonomyPredictionGenerator._yeild_geonet_predictions_and_gts(building_name))

    @staticmethod
    @lru_cache(maxsize=None)
    def get_predictions_from_geonet(building_name, dry_run=False):
        # Load from .npy if there. Else fetch below
        geonet_preds, geonet_gts, geonet_valid_pixels, geonet_image_names = [], [], [], []
        for ii, (prediction_normal, gt_normal, valid_pixels, image_name) in enumerate(
                TaskonomyPredictionGenerator.yeild_geonet_predictions_and_gts(building_name)):
            if ii % 50 == 0:
                print(ii)
            if dry_run and ii > 3:
                break

            geonet_preds.append(prediction_normal)
            geonet_gts.append(gt_normal)
            geonet_valid_pixels.append(valid_pixels)
            geonet_image_names.append(image_name)

        return np.asarray(geonet_preds), np.asarray(geonet_gts), np.asarray(geonet_valid_pixels), geonet_image_names


class NYUV2Dataset(data.Dataset):
    def __init__(self, resize_frame=(256, 256)):
        self.image_paths = []
        self.resize_frame = resize_frame
        self.nyu_norm_gt_dataset = h5py.File('/home/ajaysohmshetty/geonet/data/norm_gt_l.mat', 'r')['norm_gt_l']
        self.nyu_test_iis = scipy.io.loadmat('/home/ajaysohmshetty/geonet/data/NYU_v2_splits/test.mat')['img_set'][:,
                            0] - 1
        self.nyu_mask_dataset = h5py.File('/home/ajaysohmshetty/masks.mat', 'r')['masks']

    def __len__(self):
        return len(self.nyu_test_iis)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):  # return gt_image, normal_image, image_name
        ii = self.nyu_test_iis[index]

        geonet_nyu_output_dir = Path("/home/ajaysohmshetty/scaling/mount/shared/baseline_outputs/geonet/nyu_v2/")
        rgb_output_png = geonet_nyu_output_dir / f"{ii}_rgb_input.npy"
        rgb_image = (np.load(rgb_output_png) / 255)
        rgb_image = resize(rgb_image, self.resize_frame)
        rgb_image = rgb_image.transpose(2, 0, 1)

        normal_gt = self.nyu_norm_gt_dataset[ii]
        normal_gt = normal_gt.transpose(2, 1, 0)
        normal_gt = resize(normal_gt, self.resize_frame)
        normal_gt = convert_normal_for_display(normal_gt, should_invert=True)
        normal_gt = normal_gt.transpose(2, 0, 1)

        mask_image = self.nyu_mask_dataset[ii]
        mask_image = mask_image.transpose(1, 0)
        mask_image = np.round(resize(mask_image, self.resize_frame))

        return torch.Tensor(rgb_image), torch.Tensor(normal_gt), torch.Tensor(mask_image), torch.Tensor([ii])


class NYUV2PredictionGenerator:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_predictions_from(model_name, resize_frame=(256, 256), dry_run=False):
        nyuv2_test_dataset = NYUV2Dataset((256, 256))
        data_loader = data.DataLoader(nyuv2_test_dataset, batch_size=16)
        return TaskonomyPredictionGenerator.get_predictions_from(model_name, None, dry_run,
                                                                 custom_data_loader=data_loader)

    @staticmethod
    def yield_geonet_predictions_and_gts():
        geonet_gt_iis = []
        nyu_norm_gt_dataset = h5py.File('/home/ajaysohmshetty/geonet/data/norm_gt_l.mat', 'r')['norm_gt_l']
        geonet_nyu_output_dir = Path("/home/ajaysohmshetty/scaling/mount/shared/baseline_outputs/geonet/nyu_v2/")
        nyu_mask_dataset = h5py.File('/home/ajaysohmshetty/masks.mat', 'r')['masks']

        nyu_test_iis = scipy.io.loadmat('/home/ajaysohmshetty/geonet/data/NYU_v2_splits/test.mat')['img_set'][:, 0] - 1
        print('loaded GT normals dataset', len(nyu_test_iis))

        for ii in nyu_test_iis:
            geonet_pred_path = geonet_nyu_output_dir / f'{ii}_normals_pred.npy'

            if not geonet_pred_path.exists():
                print(f"No prediction found for {ii}, skipping...")
                continue

            prediction_normal = np.load(geonet_pred_path)
            prediction_normal = convert_normal_for_display(prediction_normal)
            # Geonet adds an extra 1 col and 1 row. IDK WHY
            prediction_normal = prediction_normal[:-1, :-1, :]

            gt_normal = nyu_norm_gt_dataset[ii]
            gt_normal = gt_normal.transpose(0, 2, 1)
            gt_normal = convert_normal_for_display(gt_normal)

            mask_image = nyu_mask_dataset[ii]
            mask_image = mask_image.transpose(1, 0)
            # mask_image = np.round(resize(mask_image, self.resize_frame))

            yield prediction_normal, gt_normal, mask_image, str(ii)

    @staticmethod
    @lru_cache(maxsize=None)
    def get_predictions_from_geonet(dry_run=False):
        # Load from .npy if there. Else fetch below
        geonet_preds, geonet_gts, geonet_masks, geonet_image_names = [], [], [], []
        for ii, (prediction_normal, gt_normal, mask_image, image_name) in enumerate(
                NYUV2PredictionGenerator.yield_geonet_predictions_and_gts()):
            if ii % 50 == 0:
                print(ii)
            if dry_run and ii > 3:
                break

            geonet_preds.append(prediction_normal)
            geonet_gts.append(gt_normal)
            geonet_masks.append(mask_image)
            geonet_image_names.append(image_name)

        return np.asarray(geonet_preds), np.asarray(geonet_gts), np.asarray(geonet_masks), geonet_image_names


def evaluation_our_models_and_geonet_on_taskonomy():
    all_models_on_nyu_metrics = {}

    for model_name, _ in our_best_models:
        print(model_name)
        if model_name == 'geonet':
            all_predictions, all_targets, all_valid_pixels, _ = NYUV2PredictionGenerator.get_predictions_from_geonet(
                dry_run=False)
        else:
            all_predictions, all_targets, all_valid_pixels, _ = NYUV2PredictionGenerator.get_predictions_from(
                model_name,
                dry_run=False)
        if model_name == 'rgb2normal_multitask.pth':
            all_predictions = all_predictions[:, :, :, -3:]

        metrics = get_metrics(all_predictions, all_targets, all_valid_pixels)
        all_models_on_nyu_metrics[model_name] = metrics

    all_models_on_taskonomy_metrics = {}

    for model_name, _ in [('geonet', None)] + our_best_models:
        print(model_name)
        if model_name == 'geonet':
            all_predictions, all_targets, all_valid_pixels, _ = TaskonomyPredictionGenerator.get_predictions_from_geonet(
                building_name='almena',
                dry_run=False)
        else:
            all_predictions, all_targets, all_valid_pixels, _ = TaskonomyPredictionGenerator.get_predictions_from(
                model_name,
                building_name='almena',
                dry_run=False)
        if model_name == 'rgb2normal_multitask.pth':
            all_predictions = all_predictions[:, :, :, -3:]

        metrics = get_metrics(all_predictions, all_targets, all_valid_pixels)
        all_models_on_taskonomy_metrics[model_name] = metrics

    return all_models_on_nyu_metrics, all_models_on_taskonomy_metrics


def evaluation_pix2pix_taskonomy():
    pix2pix_preds = []
    pix2pix_targets = []

    for ii, target_png in enumerate(Path('/home/ajaysohmshetty/scaling/mount/images').glob('*_real_B.png')):
        if ii % 100 == 0:
            print(ii)

        image_name = target_png.stem.split('_real_B')[0]
        pred_png = target_png.parents[0] / f'{image_name}_fake_B.png'

        pred_image = np.asarray(imageio.imread(str(pred_png))) / 255 * 2 - 1
        target_image = np.asarray(imageio.imread(str(target_png))) / 255 * 2 - 1

        pix2pix_preds.append(pred_image)
        pix2pix_targets.append(target_image)

    return get_metrics(np.asarray(pix2pix_preds[:1600]), np.asarray(pix2pix_targets[:1600]), np.ones((1600, 256, 256)))


our_best_models = [
    ('rgb2normal_multitask.pth', UNet(in_channels=3, out_channels=6, downsample=5)),
    ('rgb2normal_imagepercep.pth', UNet()),
    ('rgb2normal_discriminator.pth', UNet()),
    ('rgb2normal_random.pth', UNet()),
    ('unet_percep_epoch150_100.pth', UNetOld()),
    ('unet_percepstep_0.1.pth', UNetOld()),
    ('alpha_perceptriangle_method1_curve2depth.pth', UNetOld()),
    ('alpha_perceptriangle_method2_curve2depth.pth', UNetOld()),
    ('mixing_percepcurv_norm.pth', UNetOld()),
    ('unet_baseline.pth', UNetOld()),
    ('geonet', None),
]
our_best_models_to_model = {model_name: arch for model_name, arch in our_best_models}


class EvaluationMetrics:
    # Need to add to the above "our_best_models" if you want to evaluate a new pretrained model
    def run_on_model(self, model_name: str, dry_run=False):
        all_predictions, all_targets, all_valid_pixels, _ = TaskonomyPredictionGenerator.get_predictions_from(
            model_name,
            building_name='almena',
            dry_run=dry_run)
        metrics = get_metrics(all_predictions, all_targets, all_valid_pixels)
        print('Metrics for taskonomy:')
        print(metrics)
        print()

        all_predictions, all_targets, all_valid_pixels, _ = NYUV2PredictionGenerator.get_predictions_from(
            model_name,
            dry_run=False)
        metrics = get_metrics(all_predictions, all_targets, all_valid_pixels)
        print("Metrics for NYU v2:")
        print(metrics)
        print()

    # This takes a long time.
    def run_on_all(self):
        all_models_on_nyu_metrics, all_models_on_taskonomy_metrics = evaluation_our_models_and_geonet_on_taskonomy()
        pix2pix_taskonomy_metrics = evaluation_pix2pix_taskonomy()


if __name__ == '__main__':
    fire.Fire(EvaluationMetrics)
