
job run --config "cycleconsistency_curvature_mse_cycle" "python -m experiments.train_perceptual_curvature --mse-weight 1.0 --cycle-weight 1.0";
job run --config "cycleconsistency_curvature_only_cycle" "python -m experiments.train_perceptual_curvature --mse-weight 0.0 --cycle-weight 1.0";
