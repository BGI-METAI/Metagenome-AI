#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 2:51 PM
# @Author  : zhangchao
# @File    : utils.py
# @Email   : zhangchao5@genomics.cn
import logging
import numpy as np
import pickle

from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score

class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.Inf

    def __call__(self, loss):
        if np.isnan(loss):
            return True

        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EvalMetrics:
    def __init__(self, label2id_path, threshold=1, step=0.01):
        self.threshold = threshold
        self.step = step
        self.label2id = pickle.load(open(label2id_path, 'rb'))

    @staticmethod
    def compute_roc(labels, preds):
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
        roc_auc = auc(fpr, tpr)

        return roc_auc

    @staticmethod
    def f_measure(real_annots, pred_annots):
        cnt = 0
        precision = 0.0
        recall = 0.0
        p_total = 0
        for i in range(len(real_annots)):
            if len(real_annots[i]) == 0:
                continue
            tp = set(real_annots[i]).intersection(set(pred_annots[i]))
            fp = set(pred_annots[i]) - tp
            fn = set(real_annots[i]) - tp
            tpn = len(tp)
            fpn = len(fp)
            fnn = len(fn)
            cnt += 1
            recall += tpn / (1.0 * (tpn + fnn))
            if len(pred_annots[i]) > 0:
                p_total += 1
                precision_x = tpn / (1.0 * (tpn + fpn))
                precision += precision_x
        recall /= cnt
        if p_total > 0:
            precision /= p_total
        fscore = 0.0
        if precision + recall > 0:
            fscore = 2 * precision * recall / (precision + recall)
        return fscore, precision, recall

    def evaluate_annotations(self, targets_np, preds_np, terms):
        fmax = 0.0
        tmax = 0.0
        precisions = []
        recalls = []
        labels = list(map(lambda x: [terms[i] for i in np.nonzero(x == 1)[0]], targets_np))
        for t in np.arange(0, self.threshold+self.step, self.step):
            preds = preds_np.copy()
            preds[preds >= t] = 1
            preds[preds != 1] = 0
            fscore, pr, rc = self.f_measure(labels, list(map(lambda x: [terms[i] for i in np.nonzero(x == 1)[0]], preds)))
            precisions.append(pr)
            recalls.append(rc)
            if fmax < fscore:
                fmax = fscore
                tmax = t
        preds = preds_np.copy()
        preds[preds >= tmax] = 1
        preds[preds != 1] = 0
        # mcc = matthews_corrcoef(targets_np.flatten(), preds.flatten())
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = auc(recalls, precisions)
        return fmax, tmax, recalls, precisions, aupr


    def __call__(self, preds, targets):
        # auc = roc_auc_score(targets.flatten(), preds.flatten())

        total_n = 0
        total_sum = 0
        for i in range(np.size(targets,1)):
            pos_n = np.sum(targets[:, i])
            if pos_n > 0 and pos_n < len(preds):
                total_n += 1
                roc_auc = self.compute_roc(targets[:, i], preds[:, i])
                total_sum += roc_auc

        avg_auc = total_sum / total_n

        terms = {v:k for k,v in self.label2id.items()}
        fmax, tmax, recalls, precisions, aupr = self.evaluate_annotations(targets, preds, terms)
        return avg_auc, fmax, tmax, aupr



def init_logger(timestamp):
    logging.basicConfig(
        format="%(name)-12s %(levelname)-8s %(message)s",
        level=logging.INFO,
        filename=f"{timestamp}.log",
    )
    return logging.getLogger(__name__)

# eval_metrics = EvalMetrics(label2id_path='/home/share/huadjyin/home/zhangkexin2/data/proteinNER/GO/swiss-prot/filter_class/label2id.pkl')
# targets=np.random.randint(0, 2, size=(100, 10))
# preds = (targets+0.01)*0.8
# auc, avg_auc, fmax, tmax, aupr, mcc, aupr1=eval_metrics(preds,targets)
# print(auc, avg_auc, fmax, tmax, aupr, mcc, aupr1)
