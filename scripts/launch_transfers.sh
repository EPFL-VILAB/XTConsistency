

#### NEW KEYPOITNS2d Jobs #####
job run --config "transferv2_normal2keypoints2d_new" "python -m experiments.train_transfer --src_task normal --dest_task keypoints2d";
job run --config "transferv2_keypoints2d2curv_new" "python -m experiments.train_transfer --src_task keypoints2d --dest_task principal_curvature";
job run --config "transferv2_curv2keypoints2d_new" "python -m experiments.train_transfer --src_task principal_curvature --dest_task keypoints2d";
job run --config "transferv2_depth2keypoints2d_new" "python -m experiments.train_transfer --src_task depth_zbuffer --dest_task keypoints2d";
job run --config "transferv2_reshade2keypoints2d_new" "python -m experiments.train_transfer --src_task reshading --dest_task keypoints2d";
job run --config "transferv2_keypoints3d2keypoints2d_new" "python -m experiments.train_transfer --src_task keypoints3d --dest_task keypoints2d";
job run --config "transferv2_keypoints2d2normal_new" "python -m experiments.train_transfer --src_task keypoints2d --dest_task normal";
job run --config "transferv2_keypoints2d2depth_new" "python -m experiments.train_transfer --src_task keypoints2d --dest_task depth_zbuffer";
job run --config "transferv2_keypoints2d2reshade_new" "python -m experiments.train_transfer --src_task keypoints2d --dest_task reshading";

job run --config "transferv2_keypoints2d2keypoints3d_new" "python -m experiments.train_transfer --src_task keypoints2d --dest_task keypoints3d";
job run --config "transferv2_keypoints2d2edges_new" "python -m experiments.train_transfer --src_task keypoints2d --dest_task sobel_edges";
job run --config "transferv2_edges2keypoints2d_new" "python -m experiments.train_transfer --src_task sobel_edges --dest_task keypoints2d";

job run --config "transferv2_edge3d2keypoints2d_new" "python -m experiments.train_transfer --src_task edge_occlusion --dest_task keypoints2d";
job run --config "transferv2_keypoints2d2edge3d_new" "python -m experiments.train_transfer --src_task keypoints2d --dest_task edge_occlusion";


#edge3d
job run --config "transferv2_normal2edge3d" "python -m experiments.train_transfer --src_task normal --dest_task edge_occlusion";
job run --config "transferv2_rgb2edge3d" "python -m experiments.train_transfer --src_task rgb --dest_task edge_occlusion";
job run --config "transferv2_curv2edge3d" "python -m experiments.train_transfer --src_task principal_curvature --dest_task edge_occlusion";
job run --config "transferv2_depth2edge3d" "python -m experiments.train_transfer --src_task depth_zbuffer --dest_task edge_occlusion";
job run --config "transferv2_reshade2edge3d" "python -m experiments.train_transfer --src_task reshading --dest_task edge_occlusion";
job run --config "transferv2_edges2edge3d" "python -m experiments.train_transfer --src_task sobel_edges --dest_task edge_occlusion";
job run --config "transferv2_keypoints3d2edge3d" "python -m experiments.train_transfer --src_task keypoints3d --dest_task edge_occlusion";
job run --config "transferv2_edge3d2normal" "python -m experiments.train_transfer --src_task edge_occlusion --dest_task normal";
job run --config "transferv2_edge3d2depth" "python -m experiments.train_transfer --src_task edge_occlusion --dest_task depth_zbuffer";
job run --config "transferv2_edge3d2curv" "python -m experiments.train_transfer --src_task edge_occlusion --dest_task principal_curvature";
job run --config "transferv2_edge3d2reshade" "python -m experiments.train_transfer --src_task edge_occlusion --dest_task reshading";
job run --config "transferv2_edge3d2edges" "python -m experiments.train_transfer --src_task edge_occlusion --dest_task sobel_edges";

job run --config "transferv2_edge3d2keypoints3d" "python -m experiments.train_transfer --src_task edge_occlusion --dest_task keypoints3d";


#semantic segment
job run --config "transferv2_rgb2segment" "python -m experiments.train_transfer --src_task rgb --dest_task segment_semantic";

job run --config "transferv2_normal2segment" "python -m experiments.train_transfer --src_task normal --dest_task segment_semantic";
job run --config "transferv2_reshade2segment" "python -m experiments.train_transfer --src_task reshading --dest_task segment_semantic";
job run --config "transferv2_depth2segment" "python -m experiments.train_transfer --src_task depth_zbuffer --dest_task segment_semantic";



# to keypoints2d
job run --config "transferv2_keypoints2d2normal" "python -m experiments.train_transfer --src_task keypoints2d --dest_task normal";
job run --config "transferv2_keypoints2d2curv" "python -m experiments.train_transfer --src_task keypoints2d --dest_task principal_curvature";
job run --config "transferv2_keypoints2d2depth" "python -m experiments.train_transfer --src_task keypoints2d --dest_task depth_zbuffer";
job run --config "transferv2_keypoints2d2reshade" "python -m experiments.train_transfer --src_task keypoints2d --dest_task reshading";
job run --config "transferv2_keypoints2d2edges" "python -m experiments.train_transfer --src_task keypoints2d --dest_task sobel_edges";


### Running as well ####
job run --config "transferv2_normal2keypoints2d" "python -m experiments.train_transfer --src_task normal --dest_task keypoints2d";
job run --config "transferv2_curv2keypoints2d" "python -m experiments.train_transfer --src_task principal_curvature --dest_task keypoints2d";
job run --config "transferv2_depth2keypoints2d" "python -m experiments.train_transfer --src_task depth_zbuffer --dest_task keypoints2d";
job run --config "transferv2_reshade2keypoints2d" "python -m experiments.train_transfer --src_task reshading --dest_task keypoints2d";
job run --config "transferv2_edges2keypoints2d" "python -m experiments.train_transfer --src_task sobel_edges --dest_task keypoints2d";
job run --config "transferv2_keypoints3d2keypoints2d" "python -m experiments.train_transfer --src_task keypoints3d --dest_task keypoints2d";

### ALREADY QUEUED ###
job run --config "transferv2_rgb2keypoints3d" "python -m experiments.train_transfer --src_task rgb --dest_task keypoints3d";
job run --config "transferv2_depth2keypoints3d" "python -m experiments.train_transfer --src_task depth_zbuffer --dest_task keypoints3d";
job run --config "transferv2_reshade2keypoints3d" "python -m experiments.train_transfer --src_task reshading --dest_task keypoints3d";
job run --config "transferv2_keypoints2d2keypoints3d" "python -m experiments.train_transfer --src_task keypoints2d --dest_task keypoints3d";
job run --config "transferv2_keypoints3d2depth" "python -m experiments.train_transfer --src_task keypoints3d --dest_task depth_zbuffer";
job run --config "transferv2_keypoints3d2reshade" "python -m experiments.train_transfer --src_task keypoints3d --dest_task reshading";
job run --config "transferv2_keypoints3d2edges" "python -m experiments.train_transfer --src_task keypoints3d --dest_task sobel_edges";
job run --config "transferv2_edges2keypoints3d" "python -m experiments.train_transfer --src_task sobel_edges --dest_task keypoints3d";


#### RUNNING #####
job run --config "transferv2_curv2reshade" "python -m experiments.train_transfer --src_task principal_curvature --dest_task reshading";
job run --config "transferv2_depth2reshade" "python -m experiments.train_transfer --src_task depth_zbuffer --dest_task reshading";
job run --config "transferv2_reshade2curv" "python -m experiments.train_transfer --src_task reshading --dest_task principal_curvature";
job run --config "transferv2_reshade2depth" "python -m experiments.train_transfer --src_task reshading --dest_task depth_zbuffer";
job run --config "transferv2_reshade2edges" "python -m experiments.train_transfer --src_task reshading --dest_task sobel_edges";
