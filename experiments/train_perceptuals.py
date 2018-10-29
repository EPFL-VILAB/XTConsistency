from fire import Fire
from logger import VisdomLogger
from models import DataParallelModel
from modules.depth_nets import UNetDepth
from modules.percep_nets import Dense1by1Net
from modules.resnet import ResNet
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint
from utils import *


# These are outside of the main() function since they're pure helpers (ie don't need logger, main vars, etc)
def get_running_means_w_std_bounds_and_legend(list_of_values):
    running_mean_and_std_bounds = []
    legend = ["Mean-STD", "Mean", "Mean+STD"]
    for ii in range(len(list_of_values)):
        mean = np.mean(list_of_values[:ii])
        std = np.std(list_of_values[:ii])

        running_mean_and_std_bounds.append([mean - std, mean, mean + std])

    return running_mean_and_std_bounds, legend


def get_running_std(list_of_values):
    return [np.std(list_of_values[:ii]) for ii in range(len(list_of_values))]


def get_running_p_coeffs(list_of_values_1, list_of_values_2):
    assert len(list_of_values_1) == len(list_of_values_2)

    pearson_coefficients = []
    for ii in range(len(list_of_values_1)):
        if ii == 0:  # covariance is undefined if there's only one datapoint
            correlation_coefficient = 0.0
        else:
            cov = np.cov(np.stack((list_of_values_1[:ii], list_of_values_2[:ii]), axis=0))[0, 1]
            std1 = np.std(list_of_values_1[:ii])
            std2 = np.std(list_of_values_2[:ii])
            correlation_coefficient = cov / (std1 * std2)

        pearson_coefficients.append(correlation_coefficient)

    return pearson_coefficients


def main(curvature_step=0, depth_step=0, should_standardize_losses=False):
    curvature_weight = 0.0
    depth_weight = 0.0

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    # MODEL
    model = DataParallelModel(ResNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5 * i + 1 for i in range(0, 80)], gamma=0.95)
    curvature_model_base = DataParallelModel.load(Dense1by1Net().cuda(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth")
    depth_model_base = DataParallelModel.load(UNetDepth().cuda(), f"{MODELS_DIR}/normal2zdepth_unet.pth")

    def depth_model(pred):
        return checkpoint(depth_model_base, pred)

    def curvature_model(pred):
        return checkpoint(curvature_model_base, pred)

    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        mse = F.mse_loss(pred * mask.float(), target * mask.float())
        curvature = F.mse_loss(curvature_model(pred) * mask.float(), curvature_model(target) * mask.float())
        depth = F.mse_loss(depth_model(pred) * mask.float(), depth_model(target) * mask.float())

        if should_standardize_losses:
            normals_loss_std = np.std(logger.data["train_mse_loss"])
            curvature_loss_std = np.std(logger.data["train_curvature_loss"])
            depth_loss_std = np.std(logger.data["train_curvature_loss"])

            final_loss = mse / normals_loss_std
            final_loss += curvature_weight * curvature / curvature_loss_std
            final_loss += depth_weight * depth / depth_loss_std
        else:
            final_loss = mse
            final_loss += curvature_weight * curvature
            final_loss += depth_weight * depth

        metrics_to_return = (mse.detach(), curvature.detach(), depth.detach())
        return final_loss, metrics_to_return

    def jointplot1(data):
        # compute running mean for every
        data = np.stack((logger.data["train_mse_loss"], logger.data["val_mse_loss"]), axis=1)
        logger.plot(data, "mse_loss", opts={"legend": ["train_mse", "val_mse"]})

        running_mean_and_std_bounds, legend = get_running_means_w_std_bounds_and_legend(logger.data["train_mse_loss"])
        logger.plot(running_mean_and_std_bounds, "mse_loss_running_mean", opts={"legend": legend})
        logger.plot(get_running_std(logger.data["train_mse_loss"]), "mse_loss_running_stds", opts={"legend": ['STD']})

    def jointplot2(data):
        data = np.stack((logger.data["train_curvature_loss"], logger.data["val_curvature_loss"]), axis=1)
        logger.plot(data, "curvature_loss", opts={"legend": ["train_curvature", "val_curvature"]})

        running_mean_and_std_bounds, legend = get_running_means_w_std_bounds_and_legend(
            logger.data["train_curvature_loss"])
        logger.plot(running_mean_and_std_bounds, "curvature_loss_running_mean", opts={"legend": legend})
        logger.plot(get_running_std(logger.data["train_curvature_loss"]), "curvature_loss_running_stds",
                    opts={"legend": ['STD']})

    def jointplot3(data):
        data = np.stack((logger.data["train_depth_loss"], logger.data["val_depth_loss"]), axis=1)
        logger.plot(data, "depth_loss", opts={"legend": ["train_depth", "val_depth"]})

        running_mean_and_std_bounds, legend = get_running_means_w_std_bounds_and_legend(logger.data["train_depth_loss"])
        logger.plot(running_mean_and_std_bounds, "depth_loss_running_mean", opts={"legend": legend})
        logger.plot(get_running_std(logger.data["train_depth_loss"]), "depth_loss_running_stds",
                    opts={"legend": ['STD']})

    def covarianceplot(data):
        covs = get_running_p_coeffs(logger.data["train_mse_loss"], logger.data["train_curvature_loss"])
        logger.plot(covs, "train_mse_curvature_running_pearson_coeffs", opts={"legend": ['Pearson Coefficient']})

        covs = get_running_p_coeffs(logger.data["train_mse_loss"], logger.data["train_depth_loss"])
        logger.plot(covs, "train_mse_depth_running_pearson_coeffs", opts={"legend": ['Pearson Coefficient']})

        covs = get_running_p_coeffs(logger.data["train_curvature_loss"], logger.data["train_depth_loss"])
        logger.plot(covs, "train_curvature_depth_running_pearson_coeffs", opts={"legend": ['Pearson Coefficient']})

    logger.add_hook(jointplot1, feature="val_mse_loss", freq=1)
    logger.add_hook(jointplot2, feature="val_curvature_loss", freq=1)
    logger.add_hook(jointplot3, feature="val_depth_loss", freq=1)
    logger.add_hook(covarianceplot, feature="val_depth_loss", freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=48)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)
    plot_images(model, logger, test_set, ood_images, mask_val=0.502,
                loss_models={"curvature": curvature_model, "depth": depth_model})

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

        # TODO(ajay) clear out logs first, before appending to this
        # Used to log losses in case we want to analyze them afterwards for whitening
        temp_logs_location = f"{BASE_DIR}/temp_logs"
        with open(f"{temp_logs_location}/log_train_mse_losses.txt", "a") as log_file:
            log_file.write(', '.join([str(dd.cpu().tolist()) for dd in mse_data]))
            log_file.write("\n")

        with open(f"{temp_logs_location}/log_train_curvature_loss.txt", "a") as log_file:
            log_file.write(', '.join([str(dd.cpu().tolist()) for dd in curvature_data]))
            log_file.write("\n")

        with open(f"{temp_logs_location}/log_train_depth_loss.txt", "a") as log_file:
            log_file.write(', '.join([str(dd.cpu().tolist()) for dd in depth_data]))
            log_file.write("\n")

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
                    loss_models={"curvature": curvature_model, "depth": depth_model})

        scheduler.step()


if __name__ == "__main__":
    Fire(main)
