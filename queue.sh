#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --nodes=1
#SBATCH --gres=gpu:4  # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=92000       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=36:30:00     # DD-HH:MM:SS
#SBATCH --tmp=1000G
#SBATCH --output unet_dar_pe_up
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


scp ~/scratch/12NN_3_128_el.pkl $SLURM_TMPDIR/data_new/semantic2d3d/
scp ~/scratch/12NN_3_128_el_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d/
scp ~/scratch/12NN_5_128_el_test.pkl $SLURM_TMPDIR/data_new/semantic2d3d/

# tar xf ~/scratch/woodscapes.tar.gz -C $SLURM_TMPDIR/data_new
scp ~/scratch/deg_stan.pkl $SLURM_TMPDIR/data_new/semantic2d3d/
# tar xf ~/scratch/imagenet.tar.gz -C $SLURM_TMPDIR/data

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

# ./train_ddrnet.sh
# ./train_ddrnet_dar.sh
# ./train_ddrnet_cube.sh

# ./train_unet.sh
./train_unet_dar.sh
# ./train_unet_dar_knn.sh
# ./train_unet_cube.sh


# ./eval_xi_swin.sh


# ./eval_unet.sh
# ./eval_unet_dar.sh
# ./eval_unet_cube.sh


# ./eval_xi_darunet_175_high.sh
