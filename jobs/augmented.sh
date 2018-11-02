
job run --config "augmented_baseline" "python -m experiments.train_augmented"
job run --config "augmented_curvaturestep0.1" "python -m experiments.train_augmented --curvature-step 0.1"
