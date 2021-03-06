# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

import paddle

from ppad.utils.logging import get_logger

logger = get_logger("ppad")

__all__ = ['AverageMeter', 'build_record', 'log_batch', 'log_epoch']


def build_record(cfg):
    record_list = [("loss", AverageMeter('loss', '7.5f')), ("lr", AverageMeter(
        'lr', 'f',
        need_avg=False)), ("batch_time", AverageMeter('batch_cost', '.5f')),
                   ("reader_time", AverageMeter('reader_cost', '.5f'))]
    record_list = OrderedDict(record_list)
    return record_list


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name='', fmt='f', need_avg=True):
        self.name = name
        self.fmt = fmt
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        if isinstance(val, paddle.Tensor):
            val = val.numpy()[0]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}'.format(self=self)

    @property
    def total_minute(self):
        return '{self.name}_sum: {s:{self.fmt}} min'.format(
            s=self.sum / 60, self=self)

    @property
    def mean(self):
        return '{self.name}_avg: {self.avg:{self.fmt}}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        return '{self.name}: {self.val:{self.fmt}}'.format(self=self)


def log_batch(metric_list, batch_id, epoch_id, total_epoch, mode, ips):
    batch_cost = str(metric_list['batch_time'].value) + ' sec,'
    reader_cost = str(metric_list['reader_time'].value) + ' sec,'

    metric_values = []
    for m in metric_list:
        if not (m == 'batch_time' or m == 'reader_time'):
            metric_values.append(metric_list[m].value)
    metric_str = ' '.join([str(v) for v in metric_values])
    epoch_str = "epoch:[{:>3d}/{:<3d}]".format(epoch_id, total_epoch)
    step_str = "{:s} step:{:<4d}".format(mode, batch_id)

    logger.info("{:s} {:s} {:s} {:s} {:s} {}".format(
        epoch_str if batch_id == 0 else epoch_str, step_str, metric_str,
        batch_cost, reader_cost, ips))


def log_epoch(metric_list, epoch, mode, ips):
    batch_cost = 'avg_' + str(metric_list['batch_time'].value) + ' sec,'
    reader_cost = 'avg_' + str(metric_list['reader_time'].value) + ' sec,'
    batch_sum = str(metric_list['batch_time'].total) + ' sec,'

    metric_values = []
    for m in metric_list:
        if not (m == 'batch_time' or m == 'reader_time'):
            metric_values.append(metric_list[m].mean)
    metric_str = ' '.join([str(v) for v in metric_values])

    end_epoch_str = "END epoch:{:<3d}".format(epoch)

    logger.info("{:s} {:s} {:s} {:s} {:s} {:s} {}".format(
        end_epoch_str, mode, metric_str, batch_cost, reader_cost, batch_sum,
        ips))
