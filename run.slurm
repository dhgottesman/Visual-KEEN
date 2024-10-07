#! /bin/sh
#SBATCH --job-name=vk_oeg_generation_6000
#SBATCH --output=output/vk_oeg_generation_6000.out
#SBATCH --error=output/vk_oeg_generation_6000.err
#SBATCH --account=gpu-research
#SBATCH --constraint="a100"
#SBATCH --partition=gpu-a100-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-omerl-g01

python /home/morg/students/gottesman3/visual-KEEN/generation.py