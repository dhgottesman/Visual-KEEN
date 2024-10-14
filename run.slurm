#! /bin/sh
#SBATCH --job-name=vk_inputs_image_subject_2000
#SBATCH --output=output/vk_inputs_image_subject_2000.out
#SBATCH --error=output/vk_inputs_image_subject_2000.err
#SBATCH --account=gpu-research
#SBATCH --constraint="h100"
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-omerl-g01

python generation.py --start 0 --end 2000
# python generation.py --start 2000 --end 4000
# python generation.py --start 4000 --end 6000
# python generation.py --start 6000 --end 8000
# python generation.py --start 8000 --end 10000
# python generation.py --start 10000 --end -1