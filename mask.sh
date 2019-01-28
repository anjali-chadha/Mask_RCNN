#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=maskRcnn #Set the job name to "JobExample1"
#SBATCH --time=00:01:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1                   #Request 1 task
#SBATCH --mem=50G                     #Request 3200MB (2.5GB) per node
#SBATCH --ntasks=6
#SBATCH --output=maskRcnn.out      #Send stdout/err to "Example1Out.[jobID]"
#SBATCH --error=maskRcnn.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu              #Request the GPU partition/queue
##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=achadha7@tamu.edu    #Send all emails to email_address

#First Executable Line
set -ex
module load Anaconda/3-5.0.0.1
source activate MaskRcnn
module load cuDNN/7.1.4.18-fosscuda-2018b
module load CUDA/9.0.176

python evaluate.py --target /scratch/user/achadha7/PyCharm/unpaired-dehaze-gan/ablation/egan_ragan_attention_tr_256_reside_its_ots_full/test_40_rtts/images \
                    --annotations /scratch/user/achadha7/reside/beta/RTTS/Annotations \
                    --gpuId 1