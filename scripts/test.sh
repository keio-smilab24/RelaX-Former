#!/bin/bash 
# for i in {14..19}; do
#   poetry run python src/main.py test --infer_model_path ./model/model_tir_0${i}.pth |& tee log/test_id096_${i}.log
# done

# poetry run python src/main.py test --infer_model_path ./model/model_tir_0${i}.pth |& tee log/test_id096_${i}.log

poetry run python src/main.py test --infer_model_path ./model/model_tir_best.pth 

# poetry run python src/main.py create_cos_similarity

