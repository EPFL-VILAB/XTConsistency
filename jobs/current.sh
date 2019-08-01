### Set 1 of experiments
job run --instance cloud-persistent2 --config "LBPSAMPLEFF_multipercep_randombatch_percepstep0.1_normalize_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --unit-mean --percep-weight 0.0 --percep-step 0.1 --random-select --update-every-batch --subset-size 10000 --early-stopping 30"

job run --instance cloud-persistent3 --config "LBPSAMPLEFF_multipercep_randombatch_percepstep0.1_standardized_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --standardize --percep-weight 0.0 --percep-step 0.1 --random-select --update-every-batch --subset-size 10000 --early-stopping 30"

job run --instance cloud5 --config "LBPSAMPLEFF_multipercep_randombatch_percepstep0.02_standardized_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --standardize --percep-weight 0.0 --percep-step 0.02 --random-select --update-every-batch --subset-size 10000 --early-stopping 30"

job run --instance cloud-persistent2 --config "LAT_multipercep8" "python -m experiments.train_percep multiperceptual_expanded --batch-size 32 --mode winrate --k 2"


job run --instance cloud-persistent2 --config "LAT_multipercep8" "python -m experiments.train_energy multiperceptual_expanded --batch-size 32 --mode winrate --k 2"

# job run --instance cloud7 --config "LBP_multipercep_randombatch_sampleff_percepstep0.05_standardized_10k" "python -m experiments.train_percep multiperceptual_expanded --batch-size 24 --mode winrate --k 2 --update-every-batch --standardize --percep-weight 0.0 --random-select --percep-step 0.05 --subset-size 10000 --early-stopping 30"

# job run --instance cloud3 --config "LBP_multipercep_randombatch_sampleff_percepstep0.01_standardized_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --update-every-batch --standardize --percep-weight 0.0 --random-select --percep-step 0.01 --subset-size 10000 --early-stopping 30"

# job run --instance cloud3 --config "LBP_multipercep_randomepoch_sampleff_percepstep0.05_standardized_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --standardize --percep-weight 0.0 --random-select --percep-step 0.05 --subset-size 10000 --early-stopping 30"

# job run --instance cloud8 --config "LBP_multipercep_latbatch_sampleff_percepstep0.05_standardized_10k" "python -m experiments.train_percep multiperceptual_expanded --batch-size 24 --mode winrate --k 2 --update-every-batch --standardize --percep-weight 0.0 --percep-step 0.05 --subset-size 10000 --early-stopping 30"

job run --instance cloud2 --config "STD_baselinev1" "python -m experiments.train_percep baseline --batch-size 64 --early-stopping 40 --max-epochs 200" --shutdown
job run --instance cloud4 --config "STD_baselinev2" "python -m experiments.train_percep baseline --batch-size 64 --early-stopping 40 --max-epochs 200" --shutdown
job run --instance cloud5 --config "STD_baselinev3" "python -m experiments.train_percep baseline --batch-size 64 --early-stopping 40 --max-epochs 200" --shutdown

job run --instance cloud2 --config "STD_baseline_cont1" "python -m experiments.train_percep baseline --batch-size 64 --early-stopping 40 --max-epochs 100 --path mount/shared/results_STD_baselinev3_2/n.pth" --shutdown
job run --instance cloud4 --config "STD_baseline_cont2" "python -m experiments.train_percep baseline --batch-size 64 --early-stopping 40 --max-epochs 100 --path mount/shared/results_STD_baselinev3_2/n.pth" --shutdown
job run --instance cloud5 --config "STD_baseline_cont3" "python -m experiments.train_percep baseline --batch-size 64 --early-stopping 40 --max-epochs 100 --path mount/shared/results_STD_baselinev3_2/n.pth" --shutdown

# job run --instance cloud-persistent2 --config "SAMPLEFF_baseline100k" "python -m experiments.train_percep baseline --batch-size 64 --subset-size 100000 --early-stopping 40 --max-epochs 200" --shutdown


job run --instance cloud5 --config "LBP_multipercep5_lr_find" "python -m experiments.train_lr_find multiperceptual --batch-size 8 --mode winrate --k 2 --standardize"


job run --instance cloud7 --config "LBP_multipercep5_winrate_standardized_upd" "python -m experiments.train_msg multiperceptual --batch-size 24 --mode winrate --k 2 --standardize"
job run --instance cloud-persistent2 --config "LBP_multipercep8_winrate_standardized_upd" "python -m experiments.train_msg multiperceptual_expanded --batch-size 24 --mode winrate --k 2 --standardize" --shutdown

job run --instance cloud4 --config "LBP_multipercep8_random_standardized" "python -m experiments.train_msg multiperceptual_expanded --batch-size 16 --mode winrate --random-select --k 2 --standardize" --shutdown

job run --instance cloud2 --config "LBP_multipercep8_winrate_standardized_depthtarget" "python -m experiments.train_msg multiperceptual_expanded_depth --batch-size 16 --mode winrate --k 2 --standardize" --shutdown

job run --instance cloud9 --config "BASELINESV2_cycle_percepstep0.01" "python -m experiments.train_percep cycle --batch-size 32 --mode curriculum --percep-step 0.01"
job run --instance cloud10 --config "BASELINESV2_cycleconsistency_percepstep0.01" "python -m experiments.train_percep cycle_consistency --batch-size 32 --mode curriculum --percep-step 0.01"

job run --instance cloud9 --config "BASELINESV2_cycle_percepstep0.01_cont" "python -m experiments.train_percep cycle --batch-size 32 --mode curriculum --percep-step 0.01 --path mount/shared/results_BASELINESV2_cycle_percepstep0.01_7/n.pth"
job run --instance cloud10 --config "BASELINESV2_cycleconsistency_percepstep0.01" "python -m experiments.train_percep cycle_consistency --batch-size 32 --mode curriculum --percep-step 0.01 --path mount/shared/results_BASELINESV2_cycleconsistency_percepstep0.01_3/n.pth"


