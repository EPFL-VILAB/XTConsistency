

job run --config "transferv3_rgb2_segment_semantic" "python -m experiments.train_transfer rgb segment_semantic"
job run --config "transferv3_segment_semantic2_class_scene" "python -m experiments.train_transfer segment_semantic class_scene"
# job run --config "transferv3_principal_curvature2_segment_semantic" "python -m experiments.train_transfer principal_curvature segment_semantic"
# job run --config "transferv3_depth2_segment_semantic" "python -m experiments.train_transfer depth_zbuffer segment_semantic"
# job run --config "transferv3_normal2_segment_semantic" "python -m experiments.train_transfer normal segment_semantic"
