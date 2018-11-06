from experiments.plotting_fns import *
from experiments.standardization_loss_fn import get_standardization_mixed_loss_fn
from fire import Fire
from logger import VisdomLogger
from models import DataParallelModel
from modules.percep_nets import Dense1by1Net
from modules.unet import UNet
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint
from utils import *
from functools import partial


def main(curvature_step=0, depth_step=0, should_standardize=False):
    curvature_weight = 0.0
    depth_weight = 0.0

    ### MODEL ###
    model = DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    print(model.forward(torch.randn(8, 3, 256, 256)).shape)
    scheduler = MultiStepLR(model.optimizer, milestones=[5 * i + 1 for i in range(0, 80)], gamma=0.95)

    curvature_model_base = DataParallelModel.load(Dense1by1Net().cuda(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth")
    depth_model_base = None  # DataParallelModel.load(UNetDepth().cuda(), f"{MODELS_DIR}/normal2zdepth_unet.pth")

    def curvature_model(pred):
        return checkpoint(curvature_model_base, pred)

    def depth_model(pred):
        return checkpoint(depth_model_base, pred)

    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        mse = F.mse_loss(pred * mask.float(), target * mask.float())
        curvature = torch.tensor(0.0, device=mse.device) if curvature_weight == 0.0 else \
            F.mse_loss(curvature_model(pred) * mask.float(), curvature_model(target) * mask.float())
        depth = torch.tensor(0.0, device=mse.device) if depth_weight == 0.0 else \
            F.mse_loss(depth_model(pred) * mask.float(), depth_model(target) * mask.float())

        return mse + curvature_weight * curvature + depth_weight * depth, (
            mse.detach(), curvature.detach(), depth.detach())

    ### LOGGING ###
    logger = VisdomLogger("train", env=JOB)
    jointplot1_w_logger = partial(jointplot1, logger=logger)
    jointplot2_w_logger = partial(jointplot2, logger=logger)
    jointplot3_w_logger = partial(jointplot3, logger=logger)
    mseplots_w_logger = partial(mseplots, logger=logger)
    curvatureplots_w_logger = partial(curvatureplots, logger=logger)
    depthplots_w_logger = partial(depthplots, logger=logger)
    covarianceplot_w_logger = partial(covarianceplot, logger=logger)

    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)
    if should_standardize:  # TODO(nikhil,rohan): do we want these plots for the non-standardization jobs as well?
        logger.add_hook(mseplots_w_logger, feature="val_mse_loss", freq=1)
        logger.add_hook(curvatureplots_w_logger, feature="val_curvature_loss", freq=1)
        logger.add_hook(depthplots_w_logger, feature="val_depth_loss", freq=1)
        logger.add_hook(covarianceplot_w_logger, feature="val_depth_loss", freq=1)
    else:
        logger.add_hook(jointplot1_w_logger, feature="val_mse_loss", freq=1)
        logger.add_hook(jointplot2_w_logger, feature="val_curvature_loss", freq=1)
        logger.add_hook(jointplot3_w_logger, feature="val_depth_loss", freq=1)

    ### DATA LOADING ###
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=48)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)

    ### TRAINING ###
    if should_standardize:
        mixed_loss = get_standardization_mixed_loss_fn(curvature_model, depth_model, logger, False, 10)


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

        plot_images(model, logger, test_set, ood_images, mask_val=0.502,
                    loss_models={"curvature": curvature_model})  # , "depth": depth_model})

        scheduler.step()


if __name__ == "__main__":
    Fire(main)
