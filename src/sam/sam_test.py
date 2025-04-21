from sam_server import SimpleSegmentAnything
from PIL import Image
import numpy as np
import time
import cv2


def main():
    segment_anything = SimpleSegmentAnything()
    segment_anything.load_model(sam_checkpoint="src/sam/sam_vit_h_4b8939.pth")

    # 画像を読み込む
    image = Image.open("data/EXTRACTED_IMGS_/1LXtFkjw3qL/0b22fa63d0f54a529c525afbf2e8bb25/id00.jpg")
    numpy = np.asarray(image)

    start = time.time()
    masks = segment_anything.mask_generator.generate(numpy)
    end = time.time()
    print("sam time", end - start)  # sam time 2.3613719940185547

    processed_masks = segment_anything.process_anns(masks)
    overlaid = segment_anything.merge_masks(image, processed_masks)

    # overlaidをcv2で保存
    cv2.imwrite("src/sam/overlaid.jpg", overlaid)


# poetry run python src/sam/sam_test.py
if __name__ == "__main__":
    main()

# FIXME: create_dataset_for_switching.pyとmoduleのimport表記が両立できていない，importのsam.を削除すれば一時的に動く
# Traceback (most recent call last):
#   File "src/sam/sam_test.py", line 1, in <module>
#     from sam_server import SimpleSegmentAnything
#   File "/src/src/sam/sam_server.py", line 12, in <module>
#     from sam.segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
# ModuleNotFoundError: No module named 'sam'
