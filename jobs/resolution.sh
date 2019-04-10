

job run --instance cloud6 --config "2FF_train_consistency_paired_resolution_gt" "python -m experiments.train_energy consistency_paired_resolution_gt --batch-size 24 --cont mount/shared/results_2FF_train_consistency_paired_resolution_gt_18/graph.pth"
job run --instance cloud2 --config "2FF_train_consistency_paired_resolution_gt_baseline" "python -m experiments.train_energy consistency_paired_resolution_gt_baseline --batch-size 24 --cont mount/shared/results_2FF_train_consistency_paired_resolution_gt_baseline_5/graph.pth"
job run --instance cloud3 --config "2FF_train_consistency_paired_resolution_cycle" "python -m experiments.train_energy consistency_paired_resolution_cycle --batch-size 24 --cont mount/shared/results_2FF_train_consistency_paired_resolution_cycle_6/graph.pth"
job run --instance cloud4 --config "2FF_train_consistency_paired_resolution_cycle_baseline" "python -m experiments.train_energy consistency_paired_resolution_cycle_baseline --batch-size 24 --cont mount/shared/results_2FF_train_consistency_paired_resolution_cycle_baseline_5/graph.pth"

job run --config "2FF_rgb2x2normals_size512_baseline" "python -m experiments.train_energy rgb2x2normals_plots_size512 --fast --visualize" --gpus 2
job run --config "2FF_rgb2x2normals_size448_baseline" "python -m experiments.train_energy rgb2x2normals_plots_size448 --fast --visualize" --gpus 2
job run --config "2FF_rgb2x2normals_size384_baseline" "python -m experiments.train_energy rgb2x2normals_plots_size384 --fast --visualize" --gpus 2
job run --config "2FF_rgb2x2normals_size320_baseline" "python -m experiments.train_energy rgb2x2normals_plots_size320 --fast --visualize" --gpus 2
job run --config "2FF_rgb2x2normals_size512_finetuned" "python -m experiments.train_energy rgb2x2normals_plots_size512 --fast --visualize --finetuned" --gpus 2
job run --config "2FF_rgb2x2normals_size448_finetuned" "python -m experiments.train_energy rgb2x2normals_plots_size448 --fast --visualize --finetuned" --gpus 2
job run --config "2FF_rgb2x2normals_size384_finetuned" "python -m experiments.train_energy rgb2x2normals_plots_size384 --fast --visualize --finetuned" --gpus 2
job run --config "2FF_rgb2x2normals_size320_finetuned" "python -m experiments.train_energy rgb2x2normals_plots_size320 --fast --visualize --finetuned" --gpus 2