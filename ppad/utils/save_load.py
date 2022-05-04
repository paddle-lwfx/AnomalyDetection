# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import paddle
from ppad.utils.logging import get_logger

__all__ = ["load_model", "save_model"]


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    "be happy if some process has already created {}".format(
                        path))
            else:
                raise OSError("Failed to mkdir {}".format(path))


def load_model(config, model, optimizer=None):
    """
    load model from checkpoint or pretrained_model
    """
    logger = get_logger()
    global_config = config["Global"]
    checkpoints = global_config.get("checkpoints")
    pretrained_model = global_config.get("pretrained_model")
    last_model_dict = {}

    if checkpoints:
        checkpoints = checkpoints.replace(".pdparams", "")
        assert os.path.exists(checkpoints + ".pdparams"), \
            "The {}.pdparams does not exists!".format(checkpoints)

        # load params from trained model
        params = paddle.load(checkpoints + ".pdparams")
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                logger.warning("{} not in loaded params {} !".format(
                    key, params.keys()))
                continue
            pre_value = params[key]
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !".
                    format(key, value.shape, pre_value.shape))
        model.set_state_dict(new_state_dict)

        if optimizer is not None:
            if os.path.exists(checkpoints + ".pdopt"):
                optim_dict = paddle.load(checkpoints + ".pdopt")
                optimizer.set_state_dict(optim_dict)
            else:
                logger.warning(
                    f"{checkpoints}.pdopt is not exists, params of optimizer is not loaded"
                )

        if os.path.exists(checkpoints + ".states"):
            last_model_dict = paddle.load(checkpoints + ".states")
            logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        load_pretrained_params(model, pretrained_model)
    else:
        logger.info("train from scratch")

    return last_model_dict


def load_pretrained_params(model, path):
    logger = get_logger()
    if path.endswith(".pdparams"):
        path = path.replace(".pdparams", "")
    assert os.path.exists(path + ".pdparams"), \
        "The {}.pdparams does not exists!".format(path)

    params = paddle.load(path + ".pdparams")
    state_dict = model.state_dict()
    new_state_dict = {}
    for k1 in params.keys():
        if k1 not in state_dict.keys():
            logger.warning("The pretrained params {} not in model".format(k1))
        else:
            if list(state_dict[k1].shape) == list(params[k1].shape):
                new_state_dict[k1] = params[k1]
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params {} {} !".
                    format(k1, state_dict[k1].shape, k1, params[k1].shape))
    model.set_state_dict(new_state_dict)
    logger.info("load pretrain successful from {}".format(path))
    return model


def save_model(model,
               optimizer,
               model_path,
               logger,
               config,
               prefix="ppad",
               metric_dict=None**kwargs):
    """
    save model to the target path
    """
    # just save model for rank=0
    if paddle.distributed.get_rank() >= 1:
        return
    os.makedirs(model_path, exist_ok=True)
    model_prefix = os.path.join(model_path, prefix)
    paddle.save(model.state_dict(), model_prefix + ".pdparams")
    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    if metric_dict is not None:
        paddle.save(metric_dict, model_prefix + ".states")
    logger.info("save best model is to {}".format(model_prefix))
