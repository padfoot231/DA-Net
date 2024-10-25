#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --nodes=1
#SBATCH --gres=gpu:4  # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=92000       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=48:30:00     # DD-HH:MM:SS
#SBATCH --tmp=1000G
#SBATCH --output ddrnet
#nvidia-info
nvidia-smi

# Prepare virtualenv
source ~/projects/def-jlalonde/prongs/rswin/bin/activate
# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.
# DarSwin_rad_2_no_unet
# pip install typing_extensions

# #copy data
mkdir $SLURM_TMPDIR/data_new/
# tar xf ~/scratch/CVRG.tar.gz -C $SLURM_TMPDIR/data
tar xf ~/scratch/stan.tar.gz -C $SLURM_TMPDIR/data_new
# tar xf ~/scratch/woodscapes.tar.gz -C $SLURM_TMPDIR/data_new

# tar xf ~/scratch/imagenet.tar.gz -C $SLURM_TMPDIR/data

# scp ~/scratch/train.pkl $SLURM_TMPDIR/data/semantic2d3d/train.pkl
# scp ~/scratch/train_sem.pkl $SLURM_TMPDIR/data/semantic2d3d/train_sem.pkl
# scp ~/scratch/val.pkl $SLURM_TMPDIR/data/semantic2d3d/val.pkl
# scp ~/scratch/val_sem.pkl $SLURM_TMPDIR/data/semantic2d3d/val_sem.pkl
# scp ~/scratch/test.pkl $SLURM_TMPDIR/data/semantic2d3d/test.pkl
# scp ~/scratch/test_sem.pkl $SLURM_TMPDIR/data/semantic2d3d/test_sem.pkl


# # scp ~/scratch/25_4_cl_new_curve_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# # scp ~/scratch/25_4_cl_new_curve_val.pkl $SLURM_TMPDIR/data/CVRG-Pano
# # scp ~/scratch/deg.pkl $SLURM_TMPDIR/data/CVRG-Pano
scp ~/scratch/deg_stan.pkl $SLURM_TMPDIR/data_new/semantic2d3d

# # scp ~/scratch/16_4_cl_new_tan_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# # scp ~/scratch/16_4_cl_new_tan_val.pkl $SLURM_TMPDIR/data/CVRG-Pano

# # scp ~/scratch/25_4_cl_new_tan_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# # scp ~/scratch/25_4_cl_new_tan_val.pkl $SLURM_TMPDIR/data/CVRG-Pano

# # scp ~/scratch/40_4_cl_new_tan_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# # scp ~/scratch/40_4_cl_new_tan_val.pkl $SLURM_TMPDIR/data/CVRG-Pano

# # scp ~/scratch/36_4_cl_new_tan_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# # scp ~/scratch/36_4_cl_new_tan_val.pkl $SLURM_TMPDIR/data/CVRG-Pano

# # scp ~/scratch/8_8_cl_new_curve_cds_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# scp ~/scratch/8_8_cl_new_curve_cds_val.pkl $SLURM_TMPDIR/data/CVRG-Pano

# scp ~/scratch/6_6_cl_new_curve_cds_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# scp ~/scratch/6_6_cl_new_curve_cds_val.pkl $SLURM_TMPDIR/data/CVRG-Pano

# scp ~/scratch/10_10_cl_new_tan_cds_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# scp ~/scratch/10_10_cl_new_tan_cds_val.pkl $SLURM_TMPDIR/data/CVRG-Pano

# scp ~/scratch/25_4_cl_new_curve_test.pkl $SLURM_TMPDIR/data/CVRG-Pano
# scp ~/scratch/25_4_cl_new_tan_test.pkl $SLURM_TMPDIR/data/CVRG-Pano


# tar xf ~/scratch/Stanford.tar.gz -C $SLURM_TMPDIR/data
# scp ~/scratch/4NN-25_4_cl_new_curve_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# scp ~/scratch/4NN-25_4_cl_new_curve_val.pkl $SLURM_TMPDIR/data/CVRG-Pano

# scp ~/scratch/4NN-36_4_cl_new_curve_train.pkl $SLURM_TMPDIR/data/CVRG-Pano
# scp ~/scratch/4NN-36_4_cl_new_curve_val.pkl $SLURM_TMPDIR/data/CVRG-Pano

# scp ~/scratch/4NN-36_4_cl_new_curve_train.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/4NN-36_4_cl_new_curve_val.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/4NN-36_4_cl_new_curve_test.pkl $SLURM_TMPDIR/data/semantic2d3d


# scp ~/scratch/12NN_2_64_concentric.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/12NN_3_64_concentric.pkl $SLURM_TMPDIR/data/semantic2d3d

# scp ~/scratch/12NN_8_128_concentric.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_5_128_concentric_fov_90_high.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# # scp ~/scratch/12NN_5_128_concentric_fov_90_rev_high.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# # scp ~/scratch/12NN_5_128_concentric_fov_90_rev_vlow.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# # scp ~/scratch/12NN_5_128_concentric_fov_175_rev_high.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# # scp ~/scratch/12NN_5_128_concentric_fov_175_rev_vlow.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_8_128_concentric_tan_175_high.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_8_128_concentric_tan_175_vlow.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_8_128_concentric_tan_90_high.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_8_128_concentric_tan_90_vlow.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_8_128_concentric_tan_175_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_8_128_concentric_tan_90_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d

# scp ~/scratch/12NN_3_128_concentric_curve.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_1_128_concentric_ecurve.pkl $SLURM_TMPDIR/data_new/semantic2d3d

# scp ~/scratch/12NN_3_128_concentric.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_6_128_concentric.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_5_128_concentric_vlow.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_5_128_concentric_wood.pkl $SLURM_TMPDIR/data_new/woodscapes


# scp ~/scratch/12NN_6_2.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/12NN_2_64_concentric_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_3_64_concentric_test.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/12NN_8_128_concentric_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_8_128_concentric_h_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_5_128_concentric_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_3_128_concentric_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_6_128_concentric_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d
# scp ~/scratch/12NN_5_128_concentric_vlow_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d

# scp ~/scratch/12NN_1_64_concentric.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/12NN_1_64_concentric_test.pkl $SLURM_TMPDIR/data/semantic2d3d

# scp ~/scratch/12NN_25_4_tan.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/12NN_25_4_tan_test.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/12NN_25_4_curve_test.pkl $SLURM_TMPDIR/data/semantic2d3d
# # 4NN-36_4_cl_new_curve_test.pkl

# scp ~/scratch//12NN_25_4.pkl $SLURM_TMPDIR/data/CVRG-Pano
# scp ~/scratch//12NN_30_4.pkl $SLURM_TMPDIR/data/CVRG-Pano

# scp ~/scratch/12NN-25_4_cl_new_curve_test.pkl $SLURM_TMPDIR/data/CVRG-Pano

pip -V

echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Start training
# ./train_unet.sh
# ./train_darunet.sh
# ./train_unet_cubemap.sh 
./train_ddrnet.sh
# ./train_ddrnet_dar.sh
# ./train_ddrnet_cube.sh
# ./eval_xi_swin.sh
# ./eval_xi_darunet_90.sh
# ./eval_xi_darunet_175_high.sh
