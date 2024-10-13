#! /bin/sh
#SBATCH --job-name=vk_qa_generation_text_image
#SBATCH --output=output/vk_qa_generation_text_image.out
#SBATCH --error=output/vk_qa_generation_text_image.err
#SBATCH --account=gpu-research
#SBATCH --constraint="h100"
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-omerl-g01

python /home/morg/students/gottesman3/visual-KEEN/generation.py