from utils import *
from functools import partial


def get_standardization_mixed_loss_fn(curvature_model, depth_model, logger, include_depth, standardization_window_size):
    return partial(mixed_loss,
                   curvature_model=curvature_model,
                   depth_model=depth_model,
                   logger=logger,
                   include_depth=include_depth,
                   standardization_window_size=standardization_window_size,
                   )


def mixed_loss(pred, target, curvature_model, depth_model, logger, include_depth, standardization_window_size):
    mask = build_mask(target.detach(), val=0.502)
    mse = F.mse_loss(pred * mask.float(), target * mask.float())
    curvature = F.mse_loss(curvature_model(pred) * mask.float(), curvature_model(target) * mask.float())
    depth = F.mse_loss(depth_model(pred) * mask.float(), depth_model(target) * mask.float())

    if "train_mse_loss" in logger.data and len(logger.data["train_mse_loss"]) >= 2:
        normals_loss_std = np.std(logger.data["train_mse_loss"][-standardization_window_size:])
        curvature_loss_std = np.std(logger.data["train_curvature_loss"][-standardization_window_size:])
        depth_loss_std = np.std(logger.data["train_curvature_loss"][-standardization_window_size:])
        final_loss = mse / float(normals_loss_std)
        final_loss += curvature / float(curvature_loss_std)
        if include_depth:
            final_loss += depth / float(depth_loss_std)
    else:
        final_loss = mse + curvature

    metrics_to_return = (mse.detach(), curvature.detach(), depth.detach())
    return final_loss, metrics_to_return

# # TODO clear out logs first, before appending to this
# # Used to log losses in case we want to analyze them afterwards for whitening
# temp_logs_location = f"{BASE_DIR}/temp_logs"
# with open(f"{temp_logs_location}/log_train_mse_losses.txt", "a") as log_file:
#     log_file.write(', '.join([str(dd.cpu().tolist()) for dd in mse_data]))
#     log_file.write("\n")
# TODO(ajay) clear out logs first, before appending to this
# Used to log losses in case we want to analyze them afterwards for whitening
# temp_logs_location = f"{BASE_DIR}/temp_logs"
# with open(f"{temp_logs_location}/log_train_mse_losses.txt", "a") as log_file:
#     log_file.write(', '.join([str(dd.cpu().tolist()) for dd in mse_data]))
#     log_file.write("\n")

# with open(f"{temp_logs_location}/log_train_curvature_loss.txt", "a") as log_file:
#     log_file.write(', '.join([str(dd.cpu().tolist()) for dd in curvature_data]))
#     log_file.write("\n")

# with open(f"{temp_logs_location}/log_train_depth_loss.txt", "a") as log_file:
#     log_file.write(', '.join([str(dd.cpu().tolist()) for dd in depth_data]))
#     log_file.write("\n")
