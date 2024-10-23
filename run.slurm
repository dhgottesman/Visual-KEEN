#! /bin/sh
#SBATCH --job-name=vk_eval_image_embeddings_probe
#SBATCH --output=output/vk_eval_image_embeddings_probe.out
#SBATCH --error=output/vk_eval_image_embeddings_probe.err
#SBATCH --account=gpu-research
#SBATCH --constraint="h100"
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-omerl-g01

python probe.py --hidden_layer_size 4096 --learning_rate .00005 --max_iter 200 --batch_size 8 --input_type image_avg --objective qa

# python generation.py
# python generation.py --start 2000 --end 4000
# python generation.py --start 4000 --end 6000
# python generation.py --start 6000 --end 8000
# python generation.py --start 8000 --end 10000
# python generation.py --start 10000 --end -1