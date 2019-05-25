

job run --instance cloud2 --config "2FF_train_consistency_paired_resolution_gt" "python -m experiments.train_energy consistency_paired_resolution_gt --batch-size 24 --cont mount/shared/results_2FF_train_consistency_paired_resolution_gt_18/graph.pth"
job run --instance cloud2 --config "2FF_train_consistency_paired_resolution_gt_baseline" "python -m experiments.train_energy consistency_paired_resolution_gt_baseline --batch-size 24 --cont mount/shared/results_2FF_train_consistency_paired_resolution_gt_baseline_5/graph.pth"
job run --instance cloud3 --config "2FF_train_consistency_paired_resolution_cycle" "python -m experiments.train_energy consistency_paired_resolution_cycle --batch-size 24 --cont mount/shared/results_2FF_train_consistency_paired_resolution_cycle_6/graph.pth"
job run --instance cloud4 --config "2FF_train_consistency_paired_resolution_cycle_baseline" "python -m experiments.train_energy consistency_paired_resolution_cycle_baseline --batch-size 24 --cont mount/shared/results_2FF_train_consistency_paired_resolution_cycle_baseline_5/graph.pth"

job run --instance cloud3 --config "2FF_train_consistency_paired_resolution_cycle_lowweight" "python -m experiments.train_energy consistency_paired_resolution_cycle_lowweight --batch-size 16"
job run --instance cloud4 --config "2FF_train_consistency_paired_resolution_cycle_baseline_lowweight" "python -m experiments.train_energy consistency_paired_resolution_cycle_baseline_lowweight --batch-size 16"

job run --instance cloud6 --config "RES_consistency_multiresolution_gan" "python -m experiments.train_energy_subset consistency_multiresolution_gan --batch-size 24 --fast"

job run --instance cloud9 --config "VISUALS_rgb2x2normals_baseline" "python -m experiments.plot_configs [rgb2x2normals_plots, rgb2x2normals_plots_size320]"
job run --config "VISUALS_rgb2x2normals_finetuned" "python -m experiments.train_energy rgb2x2normals_plots --fast --visualize" --gpus 2
job run --config "VISUALS_rgb2x2normals_size512_baseline" "python -m experiments.train_energy rgb2x2normals_plots_size512 --fast --visualize" --gpus 2
job run --config "VISUALS_rgb2x2normals_size448_baseline" "python -m experiments.train_energy rgb2x2normals_plots_size448 --fast --visualize" --gpus 2
job run --config "VISUALS_rgb2x2normals_size384_baseline" "python -m experiments.train_energy rgb2x2normals_plots_size384 --fast --visualize" --gpus 2
job run --config "VISUALS_rgb2x2normals_size320_baseline" "python -m experiments.train_energy rgb2x2normals_plots_size320 --fast --visualize" --gpus 2
job run --config "VISUALS_rgb2x2normals_size512_finetuned" "python -m experiments.train_energy rgb2x2normals_plots_size512 --fast --visualize --finetuned" --gpus 2
job run --config "VISUALS_rgb2x2normals_size448_finetuned" "python -m experiments.train_energy rgb2x2normals_plots_size448 --fast --visualize --finetuned" --gpus 2
job run --config "VISUALS_rgb2x2normals_size384_finetuned" "python -m experiments.train_energy rgb2x2normals_plots_size384 --fast --visualize --finetuned" --gpus 2
job run --config "VISUALS_rgb2x2normals_size320_finetuned" "python -m experiments.train_energy rgb2x2normals_plots_size320 --fast --visualize --finetuned" --gpus 2

job run --config "VISUALS_y2normals_baseline" "python -m experiments.train_energy y2normals_plots --fast --visualize" --gpus 1
job run --config "VISUALS_y2normals_finetuned" "python -m experiments.train_energy y2normals_plots --fast --visualize --finetuned" --gpus 1
job run --config "VISUALS_y2normals_size512_baseline" "python -m experiments.train_energy y2normals_plots_size512 --fast --visualize" --gpus 1
job run --config "VISUALS_y2normals_size448_baseline" "python -m experiments.train_energy y2normals_plots_size448 --fast --visualize" --gpus 1
job run --config "VISUALS_y2normals_size384_baseline" "python -m experiments.train_energy y2normals_plots_size384 --fast --visualize" --gpus 1
job run --config "VISUALS_y2normals_size320_baseline" "python -m experiments.train_energy y2normals_plots_size320 --fast --visualize" --gpus 1
job run --config "VISUALS_y2normals_size512_finetuned" "python -m experiments.train_energy y2normals_plots_size512 --fast --visualize --finetuned" --gpus 1
job run --config "VISUALS_y2normals_size448_finetuned" "python -m experiments.train_energy y2normals_plots_size448 --fast --visualize --finetuned" --gpus 1
job run --config "VISUALS_y2normals_size384_finetuned" "python -m experiments.train_energy y2normals_plots_size384 --fast --visualize --finetuned" --gpus 1
job run --config "VISUALS_y2normals_size320_finetuned" "python -m experiments.train_energy y2normals_plots_size320 --fast --visualize --finetuned" --gpus 1

job run --config "VISUALS_rgb2x_baseline" "python -m experiments.train_energy rgb2x_plots --fast --visualize" --gpus 1
job run --config "VISUALS_rgb2x_baseline_size320" "python -m experiments.train_energy rgb2x_plots_size320 --fast --visualize" --gpus 1
job run --config "VISUALS_rgb2x_baseline_size384" "python -m experiments.train_energy rgb2x_plots_size384 --fast --visualize" --gpus 1
job run --config "VISUALS_rgb2x_baseline_size448" "python -m experiments.train_energy rgb2x_plots_size448 --fast --visualize" --gpus 1
job run --config "VISUALS_rgb2x_baseline_size512" "python -m experiments.train_energy rgb2x_plots_size512 --fast --visualize" --gpus 1
job run --config "VISUALS_rgb2x_finetuned" "python -m experiments.train_energy rgb2x_plots --fast --visualize --finetuned" --gpus 1
job run --config "VISUALS_rgb2x_finetuned_size320" "python -m experiments.train_energy rgb2x_plots_size320 --fast --visualize --finetuned" --gpus 1
job run --config "VISUALS_rgb2x_finetuned_size384" "python -m experiments.train_energy rgb2x_plots_size384 --fast --visualize --finetuned" --gpus 1
job run --config "VISUALS_rgb2x_finetuned_size448" "python -m experiments.train_energy rgb2x_plots_size448 --fast --visualize --finetuned" --gpus 1
job run --config "VISUALS_rgb2x_finetuned_size512" "python -m experiments.train_energy rgb2x_plots_size512 --fast --visualize --finetuned" --gpus 1

job run --instance cloud1 --config "2FF_train_gaussianblur" "python -m experiments.train_energy consistency_paired_gaussianblur --batch-size 24 --fast"
