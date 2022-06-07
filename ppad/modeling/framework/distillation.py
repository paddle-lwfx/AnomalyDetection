import paddle

from .base import BaseFramework
from ..registry import FRAMEWORK
from ..builder import build_backbone


@FRAMEWORK.register()
class Distillation(BaseFramework):
    def __init__(self, teacher_model, student_model):
        super(Distillation, self).__init__()
        self.teacher_model = build_backbone(teacher_model)
        self.student_model = build_backbone(student_model)

    def train_step(self, data_batch, **kwargs):
        X = data_batch[0]
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)

        output_pred = self.student_model(X)
        output_real = self.teacher_model(X)

        total_loss = criterion(output_pred, output_real)
