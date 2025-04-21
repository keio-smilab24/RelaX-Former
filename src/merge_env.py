"""val unseen, test作成のため，環境をバランス良くマージ"""
import json
import random
import os


def create_bundles(envs, min_total=30, max_total=70, max_var=20):
    items = list(envs.items())
    random.shuffle(items)
    bundles = {}

    while items:
        bundle_names = []
        total = 0
        i = 0
        var = random.randint(0, max_var)

        while i < len(items):
            name, count = items[i]
            if total + count <= max_total:
                bundle_names.append(name)
                total += count
                items.pop(i)
            else:
                i += 1

            if total >= min_total + var:
                break

        if bundle_names:
            bundle_key = '-'.join(bundle_names)
            bundles[bundle_key] = total

    return bundles


# poetry run python src/merge_env.py
def main():
    statistic_path = "data/hm3d/hm3d_dataset/hm3d_env_to_num_images_with_overlap.json"
    stat_dict = None
    with open(statistic_path, "r") as json_file:
        stat_dict = json.load(json_file)

    bundles = create_bundles(stat_dict)

    os.makedirs("data/hm3d/hm3d_dataset", exist_ok=True)
    with open("data/hm3d/hm3d_dataset/hm3d_env_bundle_to_num_images_with_overlap.json", 'w') as json_file:
        json.dump(bundles, json_file, indent=4)

    print(f"total sample: {sum(bundles.values())}")


# poetry run python src/merge_env.py
if __name__ == "__main__":
    main()

# {
#     "VBzV5z6i1WS-7GAhQPFzMot-3t8DB4Uzvkt-y9hTuugGdiq-tQ5s4ShP627": 38, -> test
#     "TEEsavR23oF-W7k2QWzBrFY-7UrtFsADwob-bxsVRursffK": 42, -> val
#     "4ok3usBNeis-rsggHU7g7dh-cvZr5TUy5C5": 40, -> val
#     "mL8ThkuaVTM-CrMo8WxCyVb-X7gTkoDHViv": 32, -> test
#     "c5eTyR3Rxyh-LT9Jq6dN3Ea-h1zeeAwLh9Z-7MXmsvcQjpJ-q5QZSEeHe5g-BAbdmeyTvMZ": 34, -> val
#     "GLAQ4DNUx5U-QHhQZWdMpGJ-T6nG3E2Uui9": 42, -> test
#     "66seV3BWPoX-L53DsRRk4Ch-HaxA7YrQdEC-yr17PDCnDDW-mma8eWq3nNQ-XNeHsjL6nBB": 50, -> val
#     "Qpor2mEya8F-cYkrGrCg2kB-7Ukhou1GxYi": 32, -> val
#     "FnSn2KSrALj-bCPU9suPUw9": 34, -> test
#     "BHXhpBwSMLh-qyAac8rV8Zk-hDBqLgydy1n-RJaJt8UjXav": 30,
#     "58NLZxWBSpk-MHPLjHsuG27-bzCsHPLDztK-5cdEh9F2hJL-hyFzGGJCSYs-u1bkiGRVyu9": 34, -> val
#     "nrA1tAA17Yp-LEFTm3JecaC-fsQtJ8t3nTf": 50, -> val
#     "6D36GQHuP8H-YRUkbU5xsYj-LNg5mXe1BDj-Nfvxx8J5NCo-HMkoS756sz6-SByzJLxpRGn": 50, -> test
#     "5jp3fCRSRjc-k1cupFYWXJ6-svBbv1Pavdk-F7EAMsdDASd-kJJyRFXVpx2": 48, -> val
#     "wcojb4TFT35-ziup5kvtCCR-a8BtkwhxdRV-dHwjuKfkRUR-jgPBycuV1Jq-6s7QHgap2fW": 40, -> test
#     "rXXL6twQiWc-hkr2MGpHD6B-q3hn1WQ12rz-u8ug2rtNARf-uSKXQ5fFg6u-DYehNKdT76V": 38, -> test
#     "eF36g7L6Z9M-XB4GS9ShBRE-QaLdnwvtxbs-vBMLrTe4uLA": 48, -> test
#     "XMHNu9rRQ1y-BFRyYbPCCPE-AWUFxHEyV3T-z9YwN9M8FpG": 36, -> val
#     "uLz9jNga3kC-vd3HHTEpmyA-dVW2D7TDctW-X4qjx5vquwH-rJhMRvNn4DS": 42 -> test
# }
