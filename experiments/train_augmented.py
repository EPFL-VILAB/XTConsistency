from fire import Fire
from logger import VisdomLogger
from models import DataParallelModel
from modules.depth_nets import UNetDepth
from modules.percep_nets import Dense1by1Net
from modules.resnet import ResNet
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint
from utils import *

from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset

from modules.resnet import ResNet
from modules.percep_nets import DenseNet, Dense1by1Net, DenseKernelsNet, DeepNet, BaseNet, WideNet, PyramidNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet
from sklearn.model_selection import train_test_split
from fire import Fire



def main(curvature_step=0, depth_step=0):
    curvature_weight = 0.0
    depth_weight = 0.0

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    # MODEL
    model = DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    print (model.forward(torch.randn(8, 3, 256, 256)).shape)
    
    scheduler = MultiStepLR(model.optimizer, milestones=[5 * i + 1 for i in range(0, 80)], gamma=0.95)
    curvature_model_base = DataParallelModel.load(Dense1by1Net().cuda(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth")
    depth_model_base = DataParallelModel.load(UNetDepth().cuda(), f"{MODELS_DIR}/normal2zdepth_unet_v4.pth")

    def depth_model(pred):
        return checkpoint(depth_model_base, pred)

    def curvature_model(pred):
        return checkpoint(curvature_model_base, pred)

    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        mse = F.mse_loss(pred*mask.float(), target*mask.float())
        curvature = torch.tensor(0.0, device=mse.device) if curvature_weight == 0.0 else \
            F.mse_loss(curvature_model(pred)*mask.float(), curvature_model(target)*mask.float())
        depth = torch.tensor(0.0, device=mse.device) if depth_weight == 0.0 else \
            F.mse_loss(depth_model(pred)*mask.float(), depth_model(target)*mask.float())

        return mse + curvature_weight*curvature  + depth_weight*depth, (mse.detach(), curvature.detach(), depth.detach())

    def jointplot1(data):
        data = np.stack((logger.data["train_mse_loss"], logger.data["val_mse_loss"]), axis=1)
        logger.plot(data, "mse_loss", opts={"legend": ["train_mse", "val_mse"]})

    def jointplot2(data):
        data = np.stack((logger.data["train_curvature_loss"], logger.data["val_curvature_loss"]), axis=1)
        logger.plot(data, "curvature_loss", opts={"legend": ["train_curvature", "val_curvature"]})

    def jointplot3(data):
        data = np.stack((logger.data["train_depth_loss"], logger.data["val_depth_loss"]), axis=1)
        logger.plot(data, "depth_loss", opts={"legend": ["train_depth", "val_depth"]})

    logger.add_hook(jointplot1, feature="val_mse_loss", freq=1)
    logger.add_hook(jointplot2, feature="val_curvature_loss", freq=1)
    logger.add_hook(jointplot3, feature="val_depth_loss", freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=32, batch_transforms=random_resize)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)

    def plot():
        for resize in range(256, 512, 64):
            # DATA LOADING
            train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
                load_data("rgb", "normal", batch_size=48, resize=resize)
            preds, targets, losses, _ = model.predict_with_data(test_set)
            logger.images(preds.clamp(min=0, max=1), f"predictions_{resize}", nrow=2, resize=resize)
            ood_preds = model.predict(ood_images)
            logger.images(ood_preds, f"ood_predictions_{resize}", nrow=2, resize=resize)

            scheduler.step()

    plot()
    # TRAINING
    for epochs in range(0, 800):
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        (mse_data, curvature_data, depth_data) = model.fit_with_metrics(
            train_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("train_mse_loss", np.mean(mse_data))
        logger.update("train_curvature_loss", np.mean(curvature_data))
        logger.update("train_depth_loss", np.mean(depth_data))

        val_set = itertools.islice(val_loader, val_step)
        (mse_data, curvature_data, depth_data) = model.predict_with_metrics(
            val_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("val_mse_loss", np.mean(mse_data))
        logger.update("val_curvature_loss", np.mean(curvature_data))
        logger.update("val_depth_loss", np.mean(depth_data))

        curvature_weight += curvature_step
        depth_weight += depth_step
        logger.text(f"Increasing curvature weight: {curvature_weight}")
        logger.text(f"Increasing depth weight: {depth_weight}")

        plot()


if __name__ == "__main__":
    Fire(main)
