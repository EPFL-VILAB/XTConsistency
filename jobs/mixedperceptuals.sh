
# job run --config "group_baseline" "python -m experiments.train_perceptuals"
job run --config "nvidia_percep_curvature" "python -m experiments.train_perceptuals --curvature-step 0.01"
job run --config "nvidia_percep_curvedepth" "python -m experiments.train_perceptuals --curvature-step 0.01 --depth-step 0.005"