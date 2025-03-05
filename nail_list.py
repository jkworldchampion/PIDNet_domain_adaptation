import os

extra_train_img_dir = "data/nail/non_label"
output_file = "data/nail/list/nail/extra_train.lst"

if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

with open(output_file, "w") as f:
    for filename in os.listdir(extra_train_img_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # 이미지 파일 확장자
            img_path = os.path.join("raw/non_label", filename)
            f.write(f"{img_path}\n")