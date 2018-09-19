job run --config "train_zdepth_mixed0.05" "python -m experiments.train_perceptual_zdepth --perceptual-weight 0.05";


# job run --config "train_curvature_baseline" "python -m experiments.train_perceptual_curvature";
# job run --config "train_curvature_mixed0.05" "python -m experiments.train_perceptual_curvature --perceptual-weight 0.05";
# job run --config "train_curvature_mixed0.2" "python -m experiments.train_perceptual_curvature --perceptual-weight 0.2";
# job run --config "train_curvature_mixed0.5" "python -m experiments.train_perceptual_curvature --perceptual-weight 0.5";

# job run --config "train_curvature_step0.05" "python -m experiments.train_perceptual_curvature --weight-step 0.05";
# job run --config "train_curvature_step0.01" "python -m experiments.train_perceptual_curvature --weight-step 0.01";
# job run --config "train_curvature_step0.002" "python -m experiments.train_perceptual_curvature --weight-step 0.002";

# job run --config "train_curvature_perceptualonly" "python -m experiments.train_perceptual_curvature --mse-weight 0";
