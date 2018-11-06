
# job run --config "group_baseline" "python -m experiments.train_perceptuals"
job run --config "alpha_perceptriangle_method1_curve2depth" "python -m experiments.cycle2 --method 1 --curvature-step 0.1 --depth-step 0.01"
job run --config "alpha_perceptriangle_method2_curve2depth" "python -m experiments.cycle2 --method 2 --curvature-step 0.1 --depth-step 0.01"
job run --config "alpha_perceptriangle_method3_curve2depth" "python -m experiments.cycle2 --method 3 --curvature-step 0.1 --depth-step 0.01"

# job run --instance cloud1 --config "alpha_perceptriangle_method1_depth2curve" "python -m experiments.cycle3 --method 1 --curvature-step 0.1 --depth-step 0.01"
# job run --instance cloud2 --config "alpha_perceptriangle_method2_depth2curve" "python -m experiments.cycle3 --method 2 --curvature-step 0.1 --depth-step 0.01"
# job run --instance cloud3 --config "alpha_perceptriangle_method3_depth2curve" "python -m experiments.cycle3 --method 3 --curvature-step 0.1 --depth-step 0.01"
