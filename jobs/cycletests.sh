
job run --instance cloud1 --shutdown --config "1F_onlycycle" "python -m experiments.train_functional onlycycle --pretrained"
job run --instance cloud2 --shutdown --config "1F_onlycycle_gt" "python -m experiments.train_functional onlycycle_gt --pretrained"
job run --instance cloud5 --shutdown --config "1F_onlycycle_split" "python -m experiments.train_functional onlycycle_split --pretrained"
job run --instance cloud6 --shutdown --config "1F_grounded_curvature_cycle" "python -m experiments.train_functional grounded_curvature_cycle --mode mixing --batch-size 32"
job run --instance cloud7 --shutdown --config "1F_grounded_curvature_cycle_gt" "python -m experiments.train_functional grounded_curvature_cycle_gt --mode mixing --batch-size 32"
job run --instance cloud8 --shutdown --config "1F_grounded_curvature_cycle_split" "python -m experiments.train_functional grounded_curvature_cycle_split --mode mixing --batch-size 32"

