
# job run --config "group_baseline" "python -m experiments.train_perceptuals"
job run --config "nvidia_percep_curvedepth" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.05"
job run --config "nvidia_percep_curvedepth" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.1"
job run --config "nvidia_percep_curvedepth" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.2"
job run --config "nvidia_percep_curvedepth" "python -m experiments.train_perceptuals --curvature-step 0.1 --depth-step 0.5"