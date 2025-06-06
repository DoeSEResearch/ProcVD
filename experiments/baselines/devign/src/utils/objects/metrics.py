import pandas as pd

from .. import log as logger
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve
import numpy as np

class Metrics:
    def __init__(self, predicts, labels, scores):
        self.predicts = predicts
        self.labels = labels
        self.scores = scores
        self.transform()
        # print(self.predicts)
        # ✅ 打印预测值和对应标签
        with open("output_pred_vs_label.txt", "w") as f:
            for p, l in zip(self.predicts, self.labels):
                f.write(f"Pred: {p}\tTrue: {l}")
        # print("\n=== Predicted Labels vs True Labels ===")
        # for p, l in zip(self.predicts, self.labels):
        #     print(f"Pred: {p}\tTrue: {l}")
        # print("========================================")

    def transform(self):
        self.series = pd.Series(self.scores)
        self.predicts = self.series.apply(lambda x: 1 if x >= 0.5 else 0)
        self.predicts.reset_index(drop=True, inplace=True)

    def __str__(self):
        confusion = confusion_matrix(y_true=self.labels, y_pred=self.predicts)
        tn, fp, fn, tp = confusion.ravel()
        string = f"\nConfusion matrix: \n"
        string += f"{confusion}\n"
        string += f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
        string += '\n'.join([name + ": " + str(metric) for name, metric in self().items()])
        return string

    @staticmethod
    def calculate_vul_det_score(predictions, ground_truth, target_fpr=0.005):
        """
        Calculate the vulnerability detection score (VD-S) given a tolerable FPR.
        """
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
        valid_indices = np.where(fpr <= target_fpr)[0]
        
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
        else:
            idx = np.abs(fpr - target_fpr).argmin()
            
        chosen_threshold = thresholds[idx]
        classified_preds = [1 if pred >= chosen_threshold else 0 for pred in predictions]
        
        fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
        tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
        vds = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return vds

    def __call__(self):

        _metrics = {"Accuracy": metrics.accuracy_score(y_true=self.labels, y_pred=self.predicts),
                    "Precision": metrics.precision_score(y_true=self.labels, y_pred=self.predicts),
                    "Recall": metrics.recall_score(y_true=self.labels, y_pred=self.predicts),
                    "F-measure": metrics.f1_score(y_true=self.labels, y_pred=self.predicts),
                    "Precision-Recall AUC": metrics.average_precision_score(y_true=self.labels, y_score=self.scores),
                    "AUC": metrics.roc_auc_score(y_true=self.labels, y_score=self.scores),
                    "MCC": metrics.matthews_corrcoef(y_true=self.labels, y_pred=self.predicts),
                    # "VD-S (FPR: 0.005)": self.calculate_vul_det_score(self.scores, self.labels),
                    "VD-S (FPR: 0.005)": Metrics.calculate_vul_det_score(self.scores, self.labels),  # ✅ 注意这里

                    "Error": self.error()}

        return _metrics

    def log(self):
        excluded = ["Precision-Recall AUC", "AUC"]
        _metrics = self()
        msg = ' - '.join(
            [f"({name[:3]} {round(metric, 3)})" for name, metric in _metrics.items() if name not in excluded])

        logger.log_info('metrics', msg)

    def error(self):
        errors = [(abs(score - (1 if score >= 0.5 else 0))/score)*100 for score, label in zip(self.scores, self.labels)]

        return sum(errors)/len(errors)
