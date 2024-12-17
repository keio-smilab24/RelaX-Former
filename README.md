# Open-Vocabulary Mobile Manipulation Based on Double Relaxed Contrastive Learning with Dense Labeling

Daichi Yashima, Ryosuke Korekata and Komei Sugiura

Growing labor shortages are increasing the demand for domestic service robots (DSRs) to assist in various settings. 
In this study, we develop a DSR that transports everyday objects to specified pieces of furniture based on open-vocabulary instructions.
Our approach focuses on retrieving images of target objects and receptacles from pre-collected images of indoor environments.
For example, given an instruction “Please get the right red towel hanging on the metal towel rack and put it in the white washing machine on the left,” the DSR is expected to carry the red towel to the washing machine based on the retrieved images.
This is challenging because the correct images should be retrieved from thousands of collected images, which may include many images of similar towels and appliances.
To address this, we propose RelaX-Former, which learns diverse and robust representations from among positive, unlabeled positive, and negative samples.
We evaluated RelaX-Former on a dataset containing real-world indoor images and human annotated instructions including complex referring expressions.
The experimental results demonstrate that RelaX-Former outperformed existing baseline models across standard image retrieval metrics.
Moreover, we performed physical experiments using a DSR to evaluate the performance of our approach in a zero-shot transfer setting.
The experiments involved the DSR to carry objects to specific receptacles based on open-vocabulary instructions, achieving an overall success rate of 75%.

## Setup
```bash
git clone https://github.com/keio-smilab24/RelaX-Former.git
cd RelaX-Former
./scripts/build_docker.sh
```


## Launch and Attach to docker container
1. launch container
    ```bash
    ./scripts/launch_container.sh
    ```

2. attach to container
    ```bash
    source ./config/set_constants.sh
    docker exec -it $docker_container_name bash
    ```


## Prepare Dataset
Download the necessary files from [here](https://drive.google.com/drive/u/1/folders/1yTuMtkLqXRO50y4y3oMivp4aDd5OlhEu) and save them in the `data` directory.
Run the following script to extract the files:
```bash
# outside docker
bash ./scripts/extract_and_organize_dataset.sh
```
We expect the directory structure to be the following:
```
./data
├── EXTRACTED_IMGS_
│   ├── 17DRP5sb8fy
│   ├── 1LXtFkjw3qL
│   ├── ...
│   └── zsNo4HB9uLZ
├── gpt4v_embeddings
│   ├── 2azQ1b91cZZ
│   ├── 2n8kARJN3HM
│   ├── ...
│   └── zsNo4HB9uLZ
├── image_llava_features_latent_full_bit
├── hm3d
│   ├── gpt3_embeddings
│   │   ├── 12e6joG2B2T
│   │   ├── 16tymPtM7uS
│   │   ├── ...
│   │   └── ZxkSUELrWtQ
│   ├── gpt4v_embeddings
│   │   ├── 12e6joG2B2T
│   │   ├── 16tymPtM7uS
│   │   ├── ...
│   │   └── ZxkSUELrWtQ
│   ├── hm3d_dataset
│   │   └── hm3d_database.json
│   ├── image_llava_features_latent_full_bit (unfreeze image_llava_features_latent_full_bit_hm3d.tar.gz)
│   ├── ver.3
│   │   └── train
│   │       ├── 1hovphK64XQ
│   │       ├── 1mCzDx3EMom
│   │       ├── ...
│   │       └── zWydhyFhvcj
│   └── ver.4
│       ├── train
│       │   ├── 1EiJpeRNEs1
│       │   ├── 1K7P6ZQS4VM
│       │   ├── ...
│       │   └── ZxkSUELrWtQ
│       └── val
│           ├── 3t8DB4Uzvkt
│           ├── 4ok3usBNeis
│           ├── ...
│           └── zt1RVoi7PcG
└── ltrpo_dataset
    └── ltrpo_database.json
```

Download the [ViT-H SAM model](https://github.com/facebookresearch/segment-anything#model-checkpoints) and save as `src/sam/sam_vit_h_4b8939.pth`.

Create dataset:
```bash
# Inside docker
poetry run python src/create_dataset_for_switching.py --np --full --full_2d --sam --clip --vit
poetry run python src/create_dataset_for_switching.py --use_np_cached --full --dataset
poetry run python src/create_dataset_for_switching_hm3d.py --np --full --full_2d --sam --clip --vit
poetry run python src/create_dataset_for_switching_hm3d.py --use_np_cached --full --dataset
```


## Train
```sh
# Inside docker
# Replace <WANDB_ID> with your desired value
./scripts/train.sh -i <WANDB_ID>
```


## Evaluation
```sh
# Inside docker
# Replace <MODEL_PATH> and <WANDB_ID> with your desired values
poetry run python src/main.py test --infer_model_path <MODEL_PATH> --log_wandb --wandb_name <WANDB_ID>
```

Expected results are as follow:
| [%] | Recall@5 |  Recall@10 |  Recall@20 |
| :--: | :--: | :--: | :--: |
| HM3D-FC | 55.4 | 76.3 | 91.6 |
| MP3D-FC | 57.0 | 72.4 | 82.5 |


## Oneshot
Since we are using ViT, CLIP image encoder, LLaVA, SAM, and SEEM to extract visual features, we need to extract the features for the oneshot task online.
For the current implementation of the oneshot task, we need to run the visual extraction server on a separate machine.
We used two machines both with a RTX Geforce 4090 GPU with 24GB of memory to obtain these features.
The following steps are necessary to setup the environment for the oneshot task:
1. Launch the visual extraction server on the second machine. Follow the setup instructions [here](https://github.com/keio-smilab24/reverie_retrieval_image_server).
2. Launch the oneshot server on the first machine.
```sh
# inside docker
export EMBED_IMAGE_SERVER_IP=172.xxx.xxx.xxx # change to the host where the visual extraction server is running
export EMBED_IMAGE_SERVER_PORT=5000 # change to the port where the visual extraction server is running
export OPENAI_API_KEY=your_openai_api_key
poetry run python src/main.py oneshot
```

If you want to run the oneshot task without the visual extraction server, you can use the following command:
```sh
# inside docker
export OPENAI_API_KEY=your_openai_api_key
poetry run python src/main.py oneshot --use_offline_data
```
This uses the pre-extracted features from the visual extraction server.
Use this in case you do not have access to the visual extraction server.

## model checkpoint

Model checkpoint is available [here](https://drive.google.com/file/d/1Qse4upeKwRDy3VPUkUo_q2q4VlQjA2Ya/view?usp=drive_link).

