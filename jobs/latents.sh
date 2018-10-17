
job run --config "latents_latent_step0.05" "python -m experiments.train_perceptual_curvature --weight-step 0.05 --latent";
job run --config "latents_preds_step0.05" "python -m experiments.train_perceptual_curvature --weight-step 0.05";
job run --config "latents_baseline" "python -m experiments.train_perceptual_curvature";
