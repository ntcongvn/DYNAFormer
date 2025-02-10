# Copyright (c) Selab-HCMUS and its affiliates.

import os
import logging
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import register_coco_instances
import json
logger = logging.getLogger(__name__)
#META

POLYPDB_INS_SEM_SEG_CATEGORIES = [
    "polyp"
]

# Polyp instance dataset
# ==== Predefined datasets and splits for PolypDB_INS ==========

_PREDEFINED_SPLITS_POLYPDB_INS_SEM_SEG={
    "polypdb_ins_sem_seg_train": ("polypdb_sem_train.json"),
    "polypdb_ins_sem_seg_val": ("polypdb_sem_val.json"),
    "polypdb_ins_sem_seg_test": ("polypdb_sem_test.json"),
}

def polyp_ins_load_sem_seg(gt_root, image_root,json_file, gt_ext="png", image_ext="jpg", json_ext="json"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id


    # Read JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Trích xuất danh sách file_name và mask_name
    
    file_names = [(image['file_name'],image['id']) for image in data['images']]
    mask_names = [(image['mask_name'],image['id']) for image in data['images']]

    input_files = sorted(
        ((os.path.join(image_root, f[0]),f[1]) for f in file_names if f[0].endswith(image_ext)),
        key=lambda x: x[1],
    )
    input_files = [item[0] for item in input_files]
    gt_files = sorted(
        ((os.path.join(gt_root, f[0]),f[1]) for f in mask_names if f[0].endswith(gt_ext)),
        key=lambda x: x[1],
    )
    gt_files = [item[0] for item in gt_files]
    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    assert len(input_files) == len(gt_files), "Directory {} and {} has {} and {} files, respectively.".format(image_root, gt_root, len(input_files), len(gt_files))
    logger.info("Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root))

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts

def register_all_polypdb_ins_semantic(root):
    for name in ["train","val","test"]:
        image_dir = root
        gt_dir = root 
        name = f"polypdb_ins_sem_seg_{name}"
        json_file=_PREDEFINED_SPLITS_POLYPDB_INS_SEM_SEG[name]
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir, z=os.path.join(root,json_file): polyp_ins_load_sem_seg(y, x, z, gt_ext="jpg", image_ext="jpg", json_ext="json")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=POLYPDB_INS_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="polyp_sem_seg",
            ignore_label=255,
        )
      
_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "../../../datasets"))
register_all_polypdb_ins_semantic(_root)

