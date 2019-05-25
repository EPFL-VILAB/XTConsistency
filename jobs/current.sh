### Set 1 of experiments
job run --instance cloud-persistent2 --config "LBPSAMPLEFF_multipercep_randombatch_percepstep0.1_normalize_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --unit-mean --percep-weight 0.0 --percep-step 0.1 --random-select --update-every-batch --subset-size 10000 --early-stopping 30"

job run --instance cloud-persistent3 --config "LBPSAMPLEFF_multipercep_randombatch_percepstep0.1_standardized_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --standardize --percep-weight 0.0 --percep-step 0.1 --random-select --update-every-batch --subset-size 10000 --early-stopping 30"

job run --instance cloud5 --config "LBPSAMPLEFF_multipercep_randombatch_percepstep0.02_standardized_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --standardize --percep-weight 0.0 --percep-step 0.02 --random-select --update-every-batch --subset-size 10000 --early-stopping 30"

job run --instance cloud-persistent4 --config "LBPSAMPLEFF_multipercep_latbatch_percepstep0.1_standardized_10k" "python -m experiments.train_percep multiperceptual --batch-size 16 --mode winrate --k 2 --standardize --percep-weight 0.0 --percep-step 0.1 --update-every-batch --subset-size 10000 --early-stopping 30"


# job run --instance cloud7 --config "LBP_multipercep_randombatch_sampleff_percepstep0.05_standardized_10k" "python -m experiments.train_percep multiperceptual_expanded --batch-size 24 --mode winrate --k 2 --update-every-batch --standardize --percep-weight 0.0 --random-select --percep-step 0.05 --subset-size 10000 --early-stopping 30"

# job run --instance cloud3 --config "LBP_multipercep_randombatch_sampleff_percepstep0.01_standardized_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --update-every-batch --standardize --percep-weight 0.0 --random-select --percep-step 0.01 --subset-size 10000 --early-stopping 30"

# job run --instance cloud3 --config "LBP_multipercep_randomepoch_sampleff_percepstep0.05_standardized_10k" "python -m experiments.train_percep multiperceptual --batch-size 32 --mode winrate --k 2 --standardize --percep-weight 0.0 --random-select --percep-step 0.05 --subset-size 10000 --early-stopping 30"

# job run --instance cloud8 --config "LBP_multipercep_latbatch_sampleff_percepstep0.05_standardized_10k" "python -m experiments.train_percep multiperceptual_expanded --batch-size 24 --mode winrate --k 2 --update-every-batch --standardize --percep-weight 0.0 --percep-step 0.05 --subset-size 10000 --early-stopping 30"

# job run --instance cloud1 --config "SAMPLEFF_baseline1m" "python -m experiments.train_percep baseline --batch-size 64 --subset-size 1000000 --early-stopping 40 --max-epochs 200" --shutdown

# job run --instance cloud-persistent2 --config "SAMPLEFF_baseline100k" "python -m experiments.train_percep baseline --batch-size 64 --subset-size 100000 --early-stopping 40 --max-epochs 200" --shutdown


job run --instance cloud5 --config "LBP_multipercep5_lr_find" "python -m experiments.train_lr_find multiperceptual --batch-size 8 --mode winrate --k 2 --standardize"


job run --instance cloud7 --config "LBP_multipercep5_winrate_standardized_upd" "python -m experiments.train_msg multiperceptual --batch-size 24 --mode winrate --k 2 --standardize"
job run --instance cloud-persistent2 --config "LBP_multipercep8_winrate_standardized_upd" "python -m experiments.train_msg multiperceptual_expanded --batch-size 24 --mode winrate --k 2 --standardize" --shutdown

job run --instance cloud4 --config "LBP_multipercep8_random_standardized" "python -m experiments.train_msg multiperceptual_expanded --batch-size 16 --mode winrate --random-select --k 2 --standardize" --shutdown

job run --instance cloud2 --config "LBP_multipercep8_winrate_standardized_depthtarget" "python -m experiments.train_msg multiperceptual_expanded_depth --batch-size 16 --mode winrate --k 2 --standardize" --shutdown

job run --instance cloud5 --config "LBPTESTS_no_opt_division" "python -m experiments.train_opt percep_curv --batch-size 48 --mode winrate --k 1 --unit-mean --max-epochs 10" --shutdown

job run --instance cloud10 --config "LBPTESTS_no_opt_standardized" "python -m experiments.train_opt percep_curv --batch-size 48 --mode winrate --k 1 --standardize --max-epochs 10" --shutdown

job run --instance cloud6 --config "LBPTESTS_opt_division" "python -m experiments.train_opt percep_curv --batch-size 48 --mode winrate --k 1 --unit-mean --max-epochs 10 --use-optimizer" --shutdown

job run --instance cloud10 --config "LBPTESTS_opt_standardized" "python -m experiments.train_opt percep_curv --batch-size 48 --mode winrate --k 1 --standardize --max-epochs 10 --use-optimizer" --shutdown

job run --instance cloud-persistent --config "BASELINES_multitask" "python -m experiments.train_msg multitask --batch-size 16 --max-epochs 10"

job run --instance cloud9 --config "BASELINES_cycle_percepstep0.02" "python -m experiments.train_percep cycle --batch-size 32 --mode curriculum --percep-step 0.02"
job run --instance cloud1 --config "BASELINES_cycleconsistency_percepstep0.02" "python -m experiments.train_percep cycle_consistency --batch-size 32 --mode curriculum --percep-step 0.02"
job run --instance cloud4 --config "BASELINES_doublecycle_percepstep0.02" "python -m experiments.train_percep doublecycle --batch-size 48 --mode curriculum --percep-step 0.02"


