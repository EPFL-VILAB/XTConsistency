job run --config "train_baseline" "python -m experiments.train_perceptual_loss";
job run --config "train_mixed0.02" "python -m experiments.train_perceptual_loss --perceptual-weight 0.02";
job run --config "train_mixed0.1" "python -m experiments.train_perceptual_loss --perceptual-weight 0.1";
job run --config "train_mixed0.25" "python -m experiments.train_perceptual_loss --perceptual-weight 0.25";
job run --config "train_convergence0.1" "python -m experiments.train_perceptual_loss --convergence-weight 0.1";
job run --config "train_convergence0.5" "python -m experiments.train_perceptual_loss --convergence-weight 0.5";
job run --config "train_perceptualonly" "python -m experiments.train_perceptual_loss --mse-weight 0 --perceptual-weight 1";

