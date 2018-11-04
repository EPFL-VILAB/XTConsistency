
# job run --config "group_baseline" "python -m experiments.train_perceptuals"
job run --config "alpha_baseline_depth0.1" "python -m experiments.train_perceptuals --depth-step 0.1"
job run --config "alpha_baseline_depth0.05" "python -m experiments.train_perceptuals --depth-step 0.05"
job run --config "alpha_baseline_depth0.02" "python -m experiments.train_perceptuals --depth-step 0.02"
job run --config "alpha_baseline_depth0.01" "python -m experiments.train_perceptuals --depth-step 0.01"

job run --config "alpha_baseline_curve0.1depth0.1" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.1"
job run --config "alpha_baseline_curve0.1depth0.05" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.05"
job run --config "alpha_baseline_curve0.1depth0.02" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.02"
job run --config "alpha_baseline_curve0.1depth0.01" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.01"