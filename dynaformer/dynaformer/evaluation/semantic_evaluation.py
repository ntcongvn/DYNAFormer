# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image
import copy
import time
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import DatasetEvaluator

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array

def evaluate_single(pred, gt):

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP = torch.Tensor([1]).cuda()

    # recall
    Recall = TP / (TP + FN)

    # Specificity or true negative rate
    Specificity = TN / (TN + FP)

    # Precision or positive predictive value
    Precision = TP / (TP + FP)

    # F1 score = Dice
    F1 = 2 * Precision * Recall / (Precision + Recall)

    # F2 score
    F2 = 5 * Precision * Recall / (4 * Precision + Recall)

    # Overall accuracy
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)

    # IoU for poly
    IoU_poly = TP / (TP + FP + FN)

    # IoU for background
    IoU_bg = TN / (TN + FP + FN)

    # mean IoU
    IoU_mean = (IoU_poly + IoU_bg) / 2.0

    return Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean

def Average(lst):
    return sum(lst) / len(lst)



class PolypDBSemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        visualize=False,
        visualize_path="./visualize",
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,

    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        self.visualize=visualize
        self.visualize_path=visualize_path
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._predictions = []
        self._bin_predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """

        if self.visualize is True and not os.path.exists(self.visualize_path): 
          os.makedirs(self.visualize_path)

        for input, output in zip(inputs, outputs):
                        
            #output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            output = output["sem_seg"].to(self._cpu_device)
            
            if self.visualize==True:
              mask_image1 = (copy.deepcopy(output).squeeze().numpy() * 255).astype(np.uint8)
              mask_image1 = Image.fromarray(mask_image1)
              mask_image1.save(self.visualize_path+'/'+input["file_name"][20:-4].replace("/","_")+".png")


            output=(output>=0.5).int()
            
            pred_bin=copy.deepcopy(output.numpy()).astype(float)
            
            
            
            #output=torch.where(output == 0, 255, output - 1)
            output=output-1
            output[output==-1]=255
            pred = np.array(output[0], dtype=int)        #predict 0: polyp, 255: _ignore_label
            
            
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)
            gt = gt.mean(axis=2)
            gt[gt<128]=0
            gt[gt>=128]=1
            
            gt_bin=copy.deepcopy(gt).astype(float)
            
            
            #gt=np.where(gt == 0, 255, gt - 1)
            gt=gt-1
            gt[gt==-1]=255
            gt = np.array(gt, dtype=int)        #predict 0: polyp, 255: _ignore_label

            #print(np.unique(pred))
            #print(np.unique(gt))
            
            pred[pred == self._ignore_label] = self._num_classes
            gt[gt == self._ignore_label] = self._num_classes

            
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            #print(self._conf_matrix)
            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
            self._bin_predictions.append({"pred":pred_bin,"gt":gt_bin})

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))

            self._bin_predictions = all_gather(self._bin_predictions)
            self._bin_predictions = list(itertools.chain(*self._bin_predictions))

            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes+1, np.nan, dtype=float)
        iou = np.full(self._num_classes+1, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal().astype(float)
        pos_gt = np.sum(self._conf_matrix, axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix, axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes+1, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal().astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix, axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix, axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)

        ####
        Recall_list=[]
        Specificity_list=[]
        Precision_list=[]
        F1_list=[]
        F2_list=[]
        ACC_overall_list=[]
        IoU_poly_list=[]
        IoU_bg_list=[]
        IoU_mean_list=[]
        print(f"Binary segmentation:{len(self._bin_predictions)} images")
        self._logger.info(f"Binary segmentation:{len(self._bin_predictions)} images")
        for img in self._bin_predictions:
          pd = img["pred"]
          gt = img["gt"]
          Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean=evaluate_single(torch.from_numpy(pd),torch.from_numpy(gt))
          Recall_list.append(Recall)
          Specificity_list.append(Specificity)
          Precision_list.append(Precision)
          F1_list.append(F1)
          F2_list.append(F2)
          ACC_overall_list.append(ACC_overall)
          IoU_poly_list.append(IoU_poly)
          IoU_bg_list.append(IoU_bg)
          IoU_mean_list.append(IoU_mean)
        print("Recall:",Average(Recall_list))
        print("Specificity:",Average(Specificity_list))
        print("Precision:",Average(Precision_list))
        print("F1:",Average(F1_list))
        print("F2:",Average(F2_list))
        print("ACC_overall:",Average(ACC_overall_list))
        print("IoU_poly:",Average(IoU_poly_list))
        print("IoU_bg:",Average(IoU_bg_list))
        print("IoU_mean:",Average(IoU_mean_list))
        self._logger.info(f"Recall:{Average(Recall_list)}")
        self._logger.info(f"Specificity:{Average(Specificity_list)}")
        self._logger.info(f"Precision:{Average(Precision_list)}")
        self._logger.info(f"F1:{Average(F1_list)}")
        self._logger.info(f"F2:{Average(F2_list)}")
        self._logger.info(f"ACC_overall:{Average(ACC_overall_list)}")
        self._logger.info(f"IoU_poly:{Average(IoU_poly_list)}")
        self._logger.info(f"IoU_bg:{Average(IoU_bg_list)}")
        self._logger.info(f"IoU_mean:{Average(IoU_mean_list)}")
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary
