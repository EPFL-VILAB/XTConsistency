
job run --instance cloud1 --config "F_delayed_GT_curvpercep_cycle_split" "python -m experiments.train_functional delayed_GT_curvpercep_cycle_split"
job run --instance cloud2 --config "F_wGTinflux_curvA_depthA" "python -m experiments.train_functional wGTinflux_curvA_depthA"
job run --instance cloud3 --config "F_wGTinflux_curvA_depthviacurvA" "python -m experiments.train_functional wGTinflux_curvA_depthviacurvA"
job run --instance cloud3 --config "F_wGTinflux_curvA_wGTinflux_curvAB_depthAB" "python -m experiments.train_functional wGTinflux_curvAB_depthAB"
job run --instance cloud3 --config "F_wGTinflux_curvA_depthviacurvA" "python -m experiments.train_functional wGTinflux_curvA_depthviacurvA"
job run --instance cloud3 --config "F_wGTinflux_curvA_depthA_trianglecurv2depth" "python -m experiments.train_functional wGTinflux_curvA_depthA_trianglecurv2depth"
job run --instance cloud3 --config "F_wGTinflux_curvA_depthA_triangledepth2curv_gt" "python -m experiments.train_functional wGTinflux_curvA_depthA_triangledepth2curv_gt"

job run --instance cloud1 --config "F_wGTinflux_A_trianglecurv2depth" "python -m experiments.train_functional wGTinflux_A_trianglecurv2depth"
job run --instance cloud2 --config "F_wGTinflux_A_trianglecurv2depth_gt" "python -m experiments.train_functional wGTinflux_A_percepcurv"
job run --instance cloud3 --config "F_wGTinflux_A_trianglecurv2depth2_gt" "python -m experiments.train_functional wGTinflux_A_trianglecurv2depth2_gt"

job run --instance cloud4 --config "F_wGTinflux_curvA_depthviacurvA" "python -m experiments.train_functional wGTinflux_curvA_depthviacurvA"
job run --instance cloud7 --config "F_wGTinflux_curvA_depthA_trianglecurv2depth" "python -m experiments.train_functional wGTinflux_curvA_depthA_trianglecurv2depth"
job run --instance cloud8 --config "F_wGTinflux_curvA_depthA_triangledepth2curv_gt" "python -m experiments.train_functional wGTinflux_curvA_depthA_triangledepth2curv_gt"
