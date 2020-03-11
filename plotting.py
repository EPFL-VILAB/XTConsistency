import numpy as np

def jointplot(logger, data, loss_type="mse_loss"):
    data = np.stack((data[f"train_{loss_type}"], data[f"val_{loss_type}"]), axis=1)
    logger.plot(data, loss_type, opts={"legend": [f"train_{loss_type}", f"val_{loss_type}"]})

def get_running_means_w_std_bounds_and_legend_on_diff_prev_time_step(list_of_list_values):
    running_mean_and_std_bounds = []
    legend = ["Mean-STD", "Mean Difference", "Mean+STD"]
    for ii, losses_in_batch_ii in enumerate(list_of_list_values):
        if ii == 0:  # there's no previous time step to compare to
            running_mean_and_std_bounds.append([0, 0, 0])
        else:
            loss_diffs = [loss_val - list_of_list_values[ii - 1][jj]
                          for jj, loss_val in enumerate(losses_in_batch_ii)]
            mean = np.mean(loss_diffs)
            std = np.std(loss_diffs)

            running_mean_and_std_bounds.append([mean - std, mean, mean + std])

    return running_mean_and_std_bounds, legend

def get_running_means_w_std_bounds_and_legend(list_of_list_values):
    running_mean_and_std_bounds = []
    legend = ["Mean-STD", "Mean", "Mean+STD"]
    for ii in range(len(list_of_list_values)):
        mean = np.mean(list_of_list_values[ii])
        std = np.std(list_of_list_values[ii])

        running_mean_and_std_bounds.append([mean - std, mean, mean + std])

    return running_mean_and_std_bounds, legend


def get_running_std(list_of_list_values):
    return [np.std(list_of_list_values[ii]) for ii in range(len(list_of_list_values))]


def get_running_p_coeffs(list_of_list_values_1, list_of_list_values_2):
    assert len(list_of_list_values_1) == len(list_of_list_values_2)

    pearson_coefficients = []
    for ii in range(len(list_of_list_values_1)):
        cov = np.cov(np.stack((list_of_list_values_1[ii],
                               list_of_list_values_2[ii]), axis=0))[0, 1]
        std1 = np.std(list_of_list_values_1[ii])
        std2 = np.std(list_of_list_values_2[ii])
        correlation_coefficient = cov / (std1 * std2)

        pearson_coefficients.append(correlation_coefficient)

    return pearson_coefficients

def mseplots(data, logger):
    data = np.stack((logger.data["train_mse_loss"], logger.data["val_mse_loss"]), axis=1)
    logger.plot(data, "mse_loss", opts={"legend": ["train_mse", "val_mse"]})

    running_mean_and_std_bounds, legend = get_running_means_w_std_bounds_and_legend(logger.data["val_mse_losses"])
    logger.plot(running_mean_and_std_bounds, "val_mse_loss_running_mean", opts={"legend": legend})
    logger.plot(get_running_std(logger.data["val_mse_losses"]), "val_mse_losses_running_stds",
                opts={"legend": ['STD']})

    running_mean_and_std_bounds_diff_prev_time_step, legend = \
        get_running_means_w_std_bounds_and_legend_on_diff_prev_time_step(logger.data["val_mse_losses"])
    logger.plot(running_mean_and_std_bounds_diff_prev_time_step, "val_mse_loss_diff_prev_step_running_mean",
                opts={"legend": legend})


def curvatureplots(data, logger):
    data = np.stack((logger.data["train_curvature_loss"], logger.data["val_curvature_loss"]), axis=1)
    logger.plot(data, "curvature_loss", opts={"legend": ["train_curvature", "val_curvature"]})

    running_mean_and_std_bounds, legend = get_running_means_w_std_bounds_and_legend(
        logger.data["val_curvature_losses"])
    logger.plot(running_mean_and_std_bounds, "val_curvature_loss_running_mean", opts={"legend": legend})
    logger.plot(get_running_std(logger.data["val_curvature_losses"]), "val_curvature_losses_running_stds",
                opts={"legend": ['STD']})

    running_mean_and_std_bounds_diff_prev_time_step, legend = \
        get_running_means_w_std_bounds_and_legend_on_diff_prev_time_step(logger.data["val_curvature_losses"])
    logger.plot(running_mean_and_std_bounds_diff_prev_time_step, "val_curvature_loss_diff_prev_step_running_mean",
                opts={"legend": legend})


def depthplots(data, logger):
    data = np.stack((logger.data["train_depth_loss"], logger.data["val_depth_loss"]), axis=1)
    logger.plot(data, "depth_loss", opts={"legend": ["train_depth", "val_depth"]})

    running_mean_and_std_bounds, legend = get_running_means_w_std_bounds_and_legend(logger.data["val_depth_losses"])
    logger.plot(running_mean_and_std_bounds, "val_depth_loss_running_mean", opts={"legend": legend})
    logger.plot(get_running_std(logger.data["val_depth_losses"]), "val_depth_losses_running_stds",
                opts={"legend": ['STD']})

    running_mean_and_std_bounds_diff_prev_time_step, legend = \
        get_running_means_w_std_bounds_and_legend_on_diff_prev_time_step(logger.data["val_depth_losses"])
    logger.plot(running_mean_and_std_bounds_diff_prev_time_step, "val_depth_loss_diff_prev_step_running_mean",
                opts={"legend": legend})


def covarianceplot(data, logger):
    covs = get_running_p_coeffs(logger.data["val_mse_losses"], logger.data["val_curvature_losses"])
    logger.plot(covs, "val_mse_curvature_running_pearson_coeffs", opts={"legend": ['Pearson Coefficient']})

    covs = get_running_p_coeffs(logger.data["val_mse_losses"], logger.data["val_depth_losses"])
    logger.plot(covs, "val_mse_depth_running_pearson_coeffs", opts={"legend": ['Pearson Coefficient']})

    covs = get_running_p_coeffs(logger.data["val_curvature_losses"], logger.data["val_depth_losses"])
    logger.plot(covs, "train_curvature_depth_running_pearson_coeffs", opts={"legend": ['Pearson Coefficient']})

    ratio_mse_curv_stds = [mse_std / curv_std for mse_std, curv_std in
                           zip(get_running_std(logger.data["val_mse_losses"]),
                               get_running_std(logger.data["val_curvature_losses"]))]
    logger.plot(ratio_mse_curv_stds, "val_mse_over_curvature_stds", opts={"legend": ['MSE / Curvature STD']})

    ratio_mse_depth_stds = [mse_std / depth_std for mse_std, depth_std in
                            zip(get_running_std(logger.data["val_mse_losses"]),
                                get_running_std(logger.data["val_depth_losses"]))]
    logger.plot(ratio_mse_depth_stds, "val_mse_over_depth_stds", opts={"legend": ['MSE / Depth STD']})
