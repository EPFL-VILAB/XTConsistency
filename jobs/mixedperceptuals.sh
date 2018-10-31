
# job run --config "group_baseline" "python -m experiments.train_perceptuals"
job run --config "percep_curve0.1_depth0.5" "python -m experiments.train_perceptuals --curvature-step 0.1"
job run --config "percep_curve0.1_depth0.5" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.05"
job run --config "percep_curve0.1depth0.1" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.1"
job run --config "percep_curve0.1depth0.2" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.2"
job run --config "percep_curve0.1depth0.5" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.5"