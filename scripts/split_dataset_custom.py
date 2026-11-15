import os
import shutil
import random

SOURCE_DIR = "dataset_custom"
OUTPUT_DIR = "dataset_custom_split"

classes = ["awake", "sleepy"]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for class_name in classes:
    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    images = os.listdir(source_class_dir)
    random.shuffle(images)

    total = len(images)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, split_images in split_data.items():
        output_class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for image_name in split_images:
            source_path = os.path.join(source_class_dir, image_name)
            output_path = os.path.join(output_class_dir, image_name)

            shutil.copy2(source_path, output_path)

    print(f"{class_name}:")
    print(f"  Train: {len(train_images)}")
    print(f"  Val  : {len(val_images)}")
    print(f"  Test : {len(test_images)}")

print("Split dataset selesai.")