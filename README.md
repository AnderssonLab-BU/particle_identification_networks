# Summary
Regarding to the paper, we provide code to achieve the goals as follows. Please cite the paper if you use any code from belows.

## Part 1. Ground truth simulation under the folder "/PINs_PYTORCH/ground_truth_simulation/":
    (./ground_truth_simulation/raw_traj_simulation_brownian_3d.py)          -> Generated 3-D trajectory with free diffusion;
    (./ground_truth_simulation/helper_image_born_wolf_single_spot.py)       -> Simulated images with single particle by Born-Wolf Point Spread Function;
    (./ground_truth_simulation/helper_image_DH_single_spot.py)              -> Simulated images with single particle by Double-Helix Point Spread Function;
    (./ground_truth_simulation/helper_image_astigmatism_single_spot.py)     -> Simulated images with single particle by Astigmatism Point Spread Function;
    (./ground_truth_simulation/interpolation_2d_born_wolf_normalization.py) -> Create interpolation for Born-Wolf PSF in 2D, which helps speed up calculation for image simulation;
    (./ground_truth_simulation/interpolation_dh_normalization.py)           -> Create interpolation for Double-Helix PSF, which helps speed up calculation for image simulation;
    (./ground_truth_simulation/initial_loc_sampling.py)                     -> Helper function used for generating trajectories of which distance between any two particles is no smaller than a constant;
    (./ground_truth_simulation/raw_image_merging_multiple_spots_sCMOS.py)   -> Merge all particles into one image.

## Part 2. Data under the folder "/PINs_PYTORCH/data/", which is for the ease of calculations:
    (./data/xcenter_unit_p512.npy)                          -> x center position of raw image (512x512 PX) under the Cartesian coordinate of which centering position is (0, 0), the unit is 1;
    (./data/ycenter_unit_p512.npy)                          -> y center position of raw image (512x512 PX) under the Cartesian coordinate of which centering position is (0, 0), the unit is 1;
    (./data/sigma_all_p512.csv)                             -> Pixel dependent readout noise for sCMOS data;
    (./data/fn_lambda_base_born_wolf_PIN_normalization.npy) -> Data storage for ease of calculation when Born-Wolf PSF is need;
    (./data/fn_lambda_base_dh_PIN_normalization.npy)        -> Data storage for ease of calculation when Double-Helix PSF is need;
    (./data/exp_x_centers_P1200.npy)                        -> x center position of experimental image (1200x1200 PX) under the Cartesian coordinate of which centering position is (0, 0), the unit is the pixel width of 0.11 um;
    (./data/exp_y_centers_P1200.npy)                        -> y center position of experimental image (1200x1200 PX) under the Cartesian coordinate of which centering position is (0, 0), the unit is the pixel width of 0.11 um.

## Part 3. main code of PINs under the folder "/PINs_PYTORCH/detection/":
    (./detection/workflow.py)                         -> main execution function for the training or evaluation process based on simulation dataset;
    (./detection/pin.py)                              -> PIN_{CNN};
    (./detection/pinmk2.py)                           -> PIN_{ResNet};
    (./detection/pinmk3.py)                           -> PIN_{FPN};
    (./detection/bwdataset.py)                        -> helper function used for importing images with Born-Wolf PSF;
    (./detection/dhdataset.py)                        -> helper function used for importing images with Double-Helix PSF;
    (./detection/astigdataset.py)                     -> helper function used for importing images with Astigmatism PSF;
    (./detection/config.py)                           -> configuration function used for training or evaluation based on simulation dataset;
    (./detection/detect_real_world_data_born_wolf.py) -> execution function used for detecting experimental dataset;
    (./detection/real_world_dataset.py)               -> configuration function used for detecting experimental dataset.

## Part 4. Analysis tools under the folder "/PINs_PYTORCH/analysis_tools/":    
    (./detection/find_heatmap_ids.py)           -> based on simulation datasets, find the heatmap position (in a predefined coordinates) of a fixed emitter;
    (./detection/find_ground_truth.py)          -> based on simulation datasets, find ground_truth of a specific particle given its position in a predefined heatmap coordinates;
    (./detection/find_center_of_subregions.py)  -> based on simulation datasets, obtain the original center position of a given subregion;
    (./detection/find_scmos_readout_noise.py)   -> based on simulation datasets, readout noise in a given subregion;
    (./detection/find_img_extraction.py)        -> based on simulation datasets, conduct image cropping and extraction;

    (./detection/experiment_find_heatmap_ids.py) -> based on experimental datasets, find the heatmap position (in a predefined coordinates) of a fixed emitter;
    (./detection/experiment_find_centers.py)     -> based on experimental datasets, obtain the original center position of a given subregion;
    (./detection/experiment_img_extraction.py)   -> based on experimental datasets, conduct image cropping and extraction.

## Part 5. execution file for PINs under the folder "/PINs_PYTORCH/experiments/":  
    (./detection/experiments/detection_eval.yaml)       -> implement evaluation;
    (./detection/experiments/detection_supercloud.yaml) -> implement training.


