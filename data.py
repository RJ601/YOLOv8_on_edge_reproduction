"""
data.py: builds the YOLO-format dataset that model.py trains on.

OUTPUT FOLDER STRUCTURE (created by build_dataset(), under yolo_dataset/)
    images/train/  images/val/  images/test/    the image files of each split
    labels/train/  labels/val/  labels/test/    one .txt label file per image
    data.yaml                                   the file passed to model.train(data=...)

WHAT data.yaml CONTAINS
    path:   absolute path of the dataset root (yolo_dataset/)
    train:  images/train   (relative to path)
    val:    images/val     (used during training to pick the best epoch)
    test:   images/test    (used only for the final evaluation)
    names:  class id -> class name: 0 bus, 1 car, 2 ferryboat, 3 person, 4 truck

HOW YOLO FINDS THE LABELS (data.yaml does NOT point to them)
    data.yaml lists only image folders. For every image path, Ultralytics swaps
    "/images/" for "/labels/" and the file extension for ".txt":
        yolo_dataset/images/train/a.jpg  ->  yolo_dataset/labels/train/a.txt
    So an image and its label file must have the same name stem and sit in
    matching split folders (this is why write_split() saves them that way).
    If the folder names or file names don't follow this pattern, or a label file
    is missing, the image is treated as having no objects, with no error shown.

LABEL FILE FORMAT (one line per object, values normalized to 0-1)
    class_id x_center y_center width height
    An empty .txt file = an image with no objects (a valid background image).
    The class ids must match the order of CLASSES, otherwise "person" is mislabeled.
"""

import shutil              # used to copy image files into the new folder structure
from pathlib import Path   # cleaner, OS-independent way to build file paths
import yaml                # used to write the data.yaml file (pip install pyyaml)

# The 5 classes used in the paper. The ORDER matters: the position in this list
# is the class_id used in the label files (bus=0, car=1, ferryboat=2, person=3, truck=4).
# Your Y labels must use the same ids, otherwise "person" will be mislabeled.
CLASSES = ["bus", "car", "ferryboat", "person", "truck"]

# Root folder where the YOLO-formatted dataset will be created.
OUT_DIR = Path("yolo_dataset")

class DATA:
    def __init__(self):
        self.classes = CLASSES

    def inspect(self):
        return None

    def process(self):
        return None

    def split(self):
        return None
    
    def write_split(self, name, image_paths, labels):
        """
        Writes ONE split (train, val or test) to disk.

        name        : "train", "val" or "test" -> becomes the subfolder name
        image_paths : list of image file paths for this split
        labels      : list of box-lists, aligned 1-to-1 with image_paths
                    (labels[i] belongs to image_paths[i])
        """

        # Build the two target folders for this split, e.g.
        #   yolo_dataset/images/train  and  yolo_dataset/labels/train
        img_dir = OUT_DIR / "images" / name
        lbl_dir = OUT_DIR / "labels" / name

        # Create them (and any missing parents). exist_ok=True means re-running
        # the script will not crash if the folders already exist.
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # zip() walks through images and their boxes together, so each image
        # is always paired with its own labels.
        for img_path, boxes in zip(image_paths, labels):
            img_path = Path(img_path)

            # Copy the image into images/<split>/ keeping its original filename.
            # (copy, not move, so your original data stays untouched.)
            shutil.copy(img_path, img_dir / img_path.name)

            # YOLO finds an image's labels by matching filenames:
            #   images/train/a.jpg  <->  labels/train/a.txt
            # img_path.stem is the filename without extension ("a").
            label_file = lbl_dir / f"{img_path.stem}.txt"

            with open(label_file, "w") as f:
                # One line per object in the image, in the format:
                #   class_id x_center y_center width height
                # An image with no objects gets an empty .txt file, which is valid
                # in YOLO (it is treated as a background/negative image).
                for cls, xc, yc, w, h in boxes:
                    # int(cls): class id must be an integer, not a float.
                    # :.6f     : keep 6 decimals, plenty of precision for 0-1 values.
                    f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


    def build_dataset(self):
        """Main function: gets the splits, writes all three, writes data.yaml."""

        # Call YOUR split function. The order of the returned values must match
        # this unpacking (images first, then labels, each as train/val/test).
        Xtrain, Xvalidation, Xtest, Ytrain, Yvalidation, Ytest = self.split()

        # Write each split to disk. The names "train", "val", "test" are the
        # folder names that data.yaml will point to below.
        self.write_split("train", Xtrain, Ytrain)
        self.write_split("val", Xvalidation, Yvalidation)
        self.write_split("test", Xtest, Ytest)

        # data.yaml is the file you pass to model.train(data=...).
        # It tells YOLO: where the dataset root is, where each split lives
        # (relative to that root), and the id -> class-name mapping.
        data_yaml = {
            "path": str(OUT_DIR.resolve()),  # absolute path to the dataset root
            "train": "images/train",         # training images (labels found automatically)
            "val": "images/val",             # validation images, used during training
            "test": "images/test",           # test images, used only for final evaluation
            # {0: "bus", 1: "car", ...} built from the CLASSES list above
            "names": {i: c for i, c in enumerate(CLASSES)},
        }

        # sort_keys=False keeps the keys in the order written above (more readable).
        with open(OUT_DIR / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, sort_keys=False)

        print(f"Dataset written to {OUT_DIR.resolve()}")