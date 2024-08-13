#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --nodes=1
#SBATCH --gres=gpu:4  # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=92000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=01:00:00     # DD-HH:MM:SS
#SBATCH --tmp=700G
#SBATCH --output DarSwin_5_no_unet
#nvidia-info
nvidia-smi

# Prepare virtualenv
source ~/projects/def-jlalonde/prongs/swin/bin/activate
# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.

# #copy data
mkdir $SLURM_TMPDIR/data_new/
# tar xf ~/scratch/CVRG.tar.gz -C $SLURM_TMPDIR/data
# tar xf ~/scratch/Stanford.tar.gz -C $SLURM_TMPDIR/data
tar xf ~/scratch/stan.tar.gz -C $SLURM_TMPDIR/data_new

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
scp ~/scratch/12NN_8_128_concentric.pkl $SLURM_TMPDIR/data_new/semantic2d3d
scp ~/scratch/12NN_5_128_concentric.pkl $SLURM_TMPDIR/data_new/semantic2d3d
scp ~/scratch/12NN_3_128_concentric.pkl $SLURM_TMPDIR/data_new/semantic2d3d

# scp ~/scratch/12NN_6_2.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/12NN_2_64_concentric_test.pkl $SLURM_TMPDIR/data/semantic2d3d
# scp ~/scratch/12NN_3_64_concentric_test.pkl $SLURM_TMPDIR/data/semantic2d3d
scp ~/scratch/12NN_8_128_concentric_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d
scp ~/scratch/12NN_5_128_concentric_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d
scp ~/scratch/12NN_3_128_concentric_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d

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
./train_darswin_5.sh