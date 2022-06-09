import paddle
import numpy as np
from sklearn.metrics import roc_curve, auc

from .base import BaseFramework
from ..registry import FRAMEWORK
from ..builder import build_backbone


@FRAMEWORK.register()
class Distillation(BaseFramework):
    def __init__(self,
                 teacher_model,
                 student_model,
                 loss_cfg=dict(
                     name="MseDirectionLoss", lamda=0.5)):
        super(Distillation, self).__init__(loss_cfg)
        self.teacher_model = build_backbone(teacher_model)
        self.student_model = build_backbone(student_model)

    def train_step(self, data_batch, **kwargs):
        X = data_batch[0]
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)

        output_pred = self.student_model(X)
        output_real = self.teacher_model(X)
        return self.loss_func(output_pred, output_real)

    def detection_test(self, test_dataloader, config):
        normal_class = config.DATASET.normal_class
        lamda = config.MODEL.loss_cfg
        dataset_name = config.DATASET.dataset_name
        direction_only = False

        if dataset_name != "mvtec":
            target_class = normal_class
        else:
            mvtec_good_dict = {'bottle': 3, 'cable': 5, 'capsule': 2, 'carpet': 2,
                               'grid': 3, 'hazelnut': 2, 'leather': 4, 'metal_nut': 3, 'pill': 5,
                               'screw': 0, 'tile': 2, 'toothbrush': 1, 'transistor': 3, 'wood': 2,
                               'zipper': 4
                               }
            target_class = mvtec_good_dict[normal_class]

        similarity_loss = paddle.nn.CosineSimilarity()
        label_score = []
        self.student_model.eval()
        for data in test_dataloader:
            X, Y = data
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            output_pred = self.student_model.forward(X)
            output_real = self.teacher_model(X)
            y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
            y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

            if direction_only:
                loss_1 = 1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1))
                loss_2 = 1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1))
                loss_3 = 1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1))
                total_loss = loss_1 + loss_2 + loss_3
            else:
                abs_loss_1 = paddle.mean((y_pred_1 - y_1) ** 2, axis=(1, 2, 3))
                loss_1 = 1 - similarity_loss(y_pred_1.reshape([y_pred_1.shape[0], -1]), y_1.reshape([y_1.shape[0], -1]))
                abs_loss_2 = paddle.mean((y_pred_2 - y_2) ** 2, axis=(1, 2, 3))
                loss_2 = 1 - similarity_loss(y_pred_2.reshape([y_pred_2.shape[0], -1]), y_2.reshape([y_2.shape[0], -1]))
                abs_loss_3 = paddle.mean((y_pred_3 - y_3) ** 2, axis=(1, 2, 3))
                loss_3 = 1 - similarity_loss(y_pred_3.reshape([y_pred_3.shape[0], -1]), y_3.reshape([y_3.shape[0], -1]))
                total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)

            label_score += list(zip(Y.detach().numpy().tolist(), total_loss.detach().numpy().tolist()))

        labels, scores = zip(*label_score)
        labels = np.array(labels)
        indx1 = labels == target_class
        indx2 = labels != target_class
        labels[indx1] = 1
        labels[indx2] = 0
        scores = np.array(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 4)
        return roc_auc
