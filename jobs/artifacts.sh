
job run --config "checkerboard_dilated_nobottle_baseline" "python -m experiments.train_perceptual_curvature";
job run --config "checkerboard_dilated_nobottle_step0.05" "python -m experiments.train_perceptual_curvature --weight-step 0.05";
