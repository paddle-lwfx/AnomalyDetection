from abc import abstractmethod

import paddle.nn as nn

from ..builder import build_loss


class BaseFramework(nn.Layer):
    def __init__(self, loss_cfg=dict(name="MseDirectionLoss", lamda=0.5)):
        super(BaseFramework, self).__init__()
        self.loss_func = build_loss(loss_cfg)
        pass

    def forward(self, data_batch, mode='infer'):
        if mode == 'train':
            return self.train_step(data_batch)
        elif mode == 'valid':
            return self.val_step(data_batch)
        elif mode == 'test':
            return self.test_step(data_batch)
        elif mode == 'infer':
            return self.infer_step(data_batch)
        else:
            raise NotImplementedError

    @abstractmethod
    def train_step(self, data_batch, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data_batch, **kwargs):
        """Validating step.
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, data_batch, **kwargs):
        """Test step.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_step(self, data_batch, **kwargs):
        """Infer step.
        """
        raise NotImplementedError
