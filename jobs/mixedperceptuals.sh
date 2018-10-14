
job run --config "current_baseline" "python -m experiments.train_perceptuals"
job run --config "current_percep_curvature" "python -m experiments.train_perceptuals --curvature-step 0.01"
job run --config "current_percep_curvedepth" "python -m experiments.train_perceptuals --curvature-step 0.01 --depth-step 0.005"