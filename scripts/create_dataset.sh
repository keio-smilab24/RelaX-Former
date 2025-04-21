#!/bin/bash

# poetry run python src/create_dataset_for_switching_hm3d.py --full --llava |& tee log/create_dataset_with_llava_latent_hm3d.log
# poetry run python src/create_dataset_for_switching.py --full --llava |& tee log/create_dataset_with_llava_latent.log

# poetry run python src/create_dataset_for_switching.py  --dataset
# poetry run python src/create_dataset_for_switching_hm3d.py --dataset

poetry run python src/create_dataset_for_switching.py  --dataset --np
poetry run python src/create_dataset_for_switching_hm3d.py  --dataset --np

# poetry run python src/create_dataset_for_switching.py  --interpolate
# poetry run python src/create_dataset_for_switching_hm3d.py  --interpolate

# poetry run python src/create_dataset_for_switching.py 
# poetry run python src/create_dataset_for_switching_hm3d.py

# poetry run python src/create_dataset_for_switching.py  --pseudo_gt
# poetry run python src/create_dataset_for_switching_hm3d.py --np --pseudo_gt

# poetry run python src/create_dataset_for_switching.py  --pseudo_gt_with_llava
# poetry run python src/create_dataset_for_switching_hm3d.py --pseudo_gt_with_llava

# poetry run python src/create_dataset_for_switching_hm3d.py  --roberta |& tee log/fix_pading_dataset_with_roberta_hm3d.log
# poetry run python src/create_dataset_for_switching.py --np --full --dataset --full_2d --vit --llava |& tee log/create_dataset_with_llava.log
# poetry run python src/create_dataset_for_switching_hm3d.py --np --full --dataset --full_2d --vit --llava |& tee log/create_dataset_hm3d_with_llava.log


# for test
# poetry run python src/create_dataset_for_switching.py --full --llava
# poetry run python src/create_dataset_for_switching.py --full --vit
# poetry run python src/create_dataset_for_switching_hm3d.py --full --vit 
