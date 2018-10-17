
job run --config "curvenet_dense_baseline" "python -m experiments.train_perceptual_curvature";
job run --config "curvenet_dense_step0.05" "python -m experiments.train_perceptual_curvature --weight-step 0.05";
