
from torchmetrics import F1Score, StructuralSimilarityIndexMeasure, AUROC, Accuracy, Specificity, Metric
from torchmetrics.classification.stat_scores import BinaryStatScores
from torchmetrics.utilities.compute import _safe_divide
from typing import Dict
from torch import Tensor, nn

__BINARY__: str = 'binary'



def _sensitivity_reduce(
        tp: Tensor,
        fp: Tensor,
        tn: Tensor,
        fn: Tensor,
) -> Tensor:
    # https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Sensitivity
    return _safe_divide(tp, tp + fn)


class _BinarySensitivity(BinaryStatScores):

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _sensitivity_reduce(tp, fp, tn, fn)


class _Sensitivity:
    """
    Just extended the torchmetrics lib, based on impl of binary specificity. other args are not needed for binary case.
    """
    def __new__(cls, threshold: float = 0.5, *args, **kwargs) -> Metric:
        return _BinarySensitivity(threshold, **kwargs)


class Metrics:
    """
    This class is a repository for accumulating performance metrics over a given epoch.
    """

    def __init__(self, device):
        self.f1_score: float = 0.0
        self.f1_score_calculator: Metric = F1Score(task=__BINARY__).to(device)

        self.sensitivity: float = 0.0
        self.sensitivity_calcualtor: Metric = _Sensitivity().to(device)

        self.specificity: float = 0.0
        self.specificity_calculator: Metric = Specificity(task=__BINARY__).to(device)

        self.accuracy: float = 0.0
        self.accuracy_calculator: Metric = Accuracy(task=__BINARY__).to(device)

        self.auc_roc: float = 0.0
        self.auc_roc_calculator: Metric = AUROC(task=__BINARY__).to(device)

        self.mean_iou: float = 0.0

        self.ssim: float = 0.0
        self.ssim_calculator: Metric = StructuralSimilarityIndexMeasure().to(device)

    def calculate(self, pred: Tensor, target: Tensor) -> None:
        self._f1_score(pred, target)
        self._sensitivity(pred, target)
        self._specificity(pred, target)
        self._accuracy(pred, target)
        self._auc_roc(pred, target)
        self._mean_iou(pred, target)
        self._ssim(pred, target)

    def get_mean_metrics(self, n: int) -> Dict[str, float]:
        n_float = float(n)
        self.f1_score /= n_float
        self.sensitivity /= n_float
        self.specificity /= n_float
        self.accuracy /= n_float
        self.auc_roc /= n_float
        self.mean_iou /= n_float
        self.ssim /= n_float
        return {
            "f1_score": self.f1_score,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "mean_iou": self.mean_iou,
            "ssim": self.ssim
        }

    def _f1_score(self, pred, target) -> None:
        self.f1_score += self.f1_score_calculator(pred, target).item()

    def _sensitivity(self, pred, target) -> None:
        self.sensitivity += self.sensitivity_calcualtor(pred, target).item()

    def _specificity(self, pred, target) -> None:
        self.specificity += self.specificity_calculator(pred, target).item()

    def _accuracy(self, pred, target) -> None:
        self.accuracy += self.accuracy_calculator(pred, target).item()

    def _auc_roc(self, pred, target) -> None:
        self.auc_roc += self.auc_roc_calculator(pred, target).item()

    def _mean_iou(self, pred, target) -> None:
        # https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy/script
        pred_bool = pred > 0.5
        targ_bool = target > 0.5
        intersection = (pred_bool & targ_bool).float().sum()  # Will be zero if Truth=0 or Prediction=0
        union = (pred_bool | targ_bool).float().sum()  # Will be zzero if both are 0
        iou = (intersection + 1e-6) / (union + 1e-6)  # We smooth our devision to avoid 0/0
        self.mean_iou += iou.item()

    def _ssim(self, pred, target) -> None:
        self.ssim += self.ssim_calculator(pred, target).item()
