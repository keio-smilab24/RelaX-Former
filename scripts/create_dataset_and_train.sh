#!/bin/bash

# poetry run python src/create_dataset_for_switching.py --np --full --dataset --full_2d --vit --llava --sam |& tee -a log/create_dataset_with_llava.log
# poetry run python src/create_dataset_for_switching_hm3d.py --np --full --dataset --full_2d --vit --llava --sam |& tee -a log/create_dataset_hm3d_with_llava.log

# poetry run python src/create_dataset_for_switching.py --np --full --dataset --full_2d --sam |& tee -a log/create_dataset_reproduce_baseline.log
# poetry run python src/create_dataset_for_switching_hm3d.py --np --full --dataset --full_2d --sam |& tee -a log/create_dataset_hm3d_reproduce_baseline.log

poetry run python src/create_dataset_for_switching.py --np 
# poetry run python src/create_dataset_for_switching_hm3d.py --np 

# poetry run python src/main.py train --eval_unseen --epoch 20 --lr 1e-4  --log_wandb --wandb_name id053 |& tee -a log/id057.log
sh ./scripts/train.sh -i id138
