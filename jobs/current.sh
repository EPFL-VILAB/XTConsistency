
job run --instance cloud3 --config "SAMPLEFF_baseline100k" "python -m experiments.train_percep baseline --batch-size 128 --subset-size 100000 --early-stopping 40 --max-epochs 200" --shutdown

# job run --instance cloud4 --config "SAMPLEFF_baseline2x100k" "python -m experiments.train_percep baseline --batch-size 128 --subset-size 200000 --early-stopping 40 --max-epochs 200" --shutdown

# job run --instance cloud3 --config "SAMPLEFF_baseline1m" "python -m experiments.train_percep baseline --batch-size 128 --subset-size 1000000 --early-stopping 40 --max-epochs 200" --shutdown

# job run --instance cloud4 --config "SAMPLEFF_baseline2x1m" "python -m experiments.train_percep baseline --batch-size 128 --subset-size 2000000 --early-stopping 40 --max-epochs 200" --shutdown

# job run --instance cloud10 --config "SAMPLEFF_consistency10k" "python -m experiments.train_consistency --subset-size 10000 --mode curriculum --percep-step 0.1 --max-epochs 200" --shutdown

# job run --instance cloud5 --config "SAMPLEFF_consistency100k" "python -m experiments.train_consistency --subset-size 100000 --mode curriculum --percep-step 0.1 --max-epochs 200" --shutdown

# job run --instance cloud7 --config "SAMPLEFF_consistency1m" "python -m experiments.train_consistency --subset-size 1000000 --mode curriculum --percep-step 0.1 --max-epochs 200" --shutdown

# job run --instance cloud4 --config "SAMPLEFF_full_data_multipercep_step0.1" "python -m experiments.train_percep multipercep --mode curriculum --batch-size 16 --max-epochs 200 --percep-step 0.1 --fast" --shutdown


