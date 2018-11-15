

job run --instance cloud1 --config "F_wGTinflux_A_percepcurv_0.1" "python -m experiments.train_functional wGTinflux_A_percepcurv --mode curriculum"

job run --instance cloud3 --config "F_wGTinflux_depthAB" "python -m experiments.train_functional wGTinflux_depthAB"
job run --instance cloud4 --config "F_wGTinflux_curvA_depthB" "python -m experiments.train_functional wGTinflux_curvA_depthB"
job run --instance cloud6 --config "F_wGTinflux_curvA_depthB_trianglecurv2depth_gt" "python -m experiments.train_functional wGTinflux_curvA_depthB_trianglecurv2depth_gt"
job run --instance cloud7 --config "F_wGTinflux_curvA_depthB_triangledepth2curv_gt" "python -m experiments.train_functional wGTinflux_curvA_depthB_triangledepth2curv_gt"
job run --instance cloud8 --config "F_wGTinflux_curvA_depthI_trianglecurv2depth_gt" "python -m experiments.train_functional wGTinflux_curvA_depthI_trianglecurv2depth_gt"
job run --instance cloud2 --config "F_wGTinflux_curvA_depthI_triangledepth2curv_gt" "python -m experiments.train_functional wGTinflux_curvA_depthI_triangledepth2curv_gt"
job run --instance cloud2 --config "F_wGTinflux_curvA_depthI_triangledepth2curv_gt" "python -m experiments.train_functional wGTinflux_curvA_depthI_triangledepth2curv_gt"
job run --config "F_wGTinflux_curvA_depthB_2dkeyptA" "python -m experiments.train_functional wGTinflux_curvA_depthB_2dkeyptA"
job run --config "F_wGTinflux_curvA_2dkeyptA" "python -m experiments.train_functional wGTinflux_curvA_2dkeyptA"

job run --config "F_wGTinflux_A_trianglecurv2depth_step" "python -m experiments.train_functional wGTinflux_A_trianglecurv2depth --mode curriculum"
job run --config "F_wGTinflux_A_trianglecurv2depth_gt_step" "python -m experiments.train_functional wGTinflux_A_trianglecurv2depth_gt --mode curriculum"
job run --config "F_wGTinflux_A_trianglecurv2depth2_gt_step" "python -m experiments.train_functional wGTinflux_A_trianglecurv2depth2_gt --mode curriculum"
