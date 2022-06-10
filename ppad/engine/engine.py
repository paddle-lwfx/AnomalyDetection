import os
import time

import paddle
from paddle.optimizer.lr import LRScheduler

from ppad.utils import get_logger, build_record, log_batch, log_epoch
from ppad.modeling import build_model
from ppad.datasets import build_dataset, build_dataloader
from ppad.optimizer import build_optimizer


def train_model(cfg, validate=False):
    logger = get_logger()
    model_name = cfg.Global.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    epochs = cfg.Global.epochs

    model = build_model(cfg.Model)

    batch_size = cfg.Dataset.batch_size
    train_dataset = build_dataset(cfg.Dataset.train)

    train_dataloader_setting = dict(
        batch_size=batch_size,
        num_workers=cfg.Dataset.num_worker,
        shuffle=True,
        drop_last=False)
    train_loader = build_dataloader(train_dataset, **train_dataloader_setting)

    if validate:
        eval_dataset = build_dataset(cfg.Dataset.eval)
        eval_dataloader_setting = dict(
            batch_size=batch_size,
            num_workers=cfg.Dataset.num_worker,
            shuffle=False,
            drop_last=False)
        eval_loader = build_dataloader(eval_dataset, **eval_dataloader_setting)

    optimizer, lr = build_optimizer(cfg.Optimizer, epochs,
                                    len(train_loader), model)
    best = 0.0
    for epoch in range(0, epochs):
        model.train()
        record_list = build_record(cfg.Model)
        tic = time.time()
        for i, data in enumerate(train_loader):
            record_list['reader_time'].update(time.time() - tic)
            loss = model(data, mode='train')
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if isinstance(lr, LRScheduler):
                lr.step()

            record_list['lr'].update(optimizer.get_lr(), batch_size)
            record_list['batch_time'].update(time.time() - tic)
            record_list['loss'].update(loss, batch_size)

            if i % cfg.get("log_interval", 10) == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / record_list["batch_time"].val)
                log_batch(record_list, i, epoch + 1, epochs, "train", ips)

            tic = time.time()

        ips = "avg_ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count /
            record_list["batch_time"].sum)
        log_epoch(record_list, epoch + 1, "train", ips)

        if validate and (epoch % cfg.get("val_interval", 1) == 0 or
                         epoch == cfg.epochs - 1):
            if cfg.Model.framework in ['KDAD']:
                eval_res = model.detection_test(eval_loader, cfg)
                logger.info(f"[Eval] epoch:{epoch} AUC: {eval_res}")
            if eval_res > best:
                best = eval_res
                paddle.save(model.state_dict(),
                            os.path.join(output_dir,
                                         f'{model_name}_best_model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(output_dir,
                                         f'{model_name}_best_model.pdopt'))
                logger.info("Already save the best model")

        if epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1:
            paddle.save(
                model.state_dict(),
                os.path.join(output_dir,
                             f'{model_name}_epoch_{epoch}_model.pdparams'))
            paddle.save(
                optimizer.state_dict(),
                os.path.join(output_dir,
                             f'{model_name}_epoch_{epoch}_model.pdopt'))
