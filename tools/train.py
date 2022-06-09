import time

from paddle.optimizer.lr import LRScheduler

from ppad.utils import get_logger, build_record, log_batch
from ppad.modeling import build_model
from ppad.datasets import build_dataset, build_dataloader
from ppad.optimizer import build_optimizer


def train_model(cfg):
    logger = get_logger()

    model = build_model(cfg.MODEL)

    batch_size = cfg.DATASET.batch_size
    train_dataset = build_dataset((cfg.DATASET.train, cfg.PIPELINE.train))

    train_dataloader_setting = dict(
        batch_size=batch_size,
        num_workers=cfg.DATASET.num_worker,
        shuffle=True,
        drop_last=False, )
    train_loader = build_dataloader(train_dataset, **train_dataloader_setting)

    optimizer, lr = build_optimizer(cfg.OPTIMIZER, cfg.epochs,
                                    len(train_loader), model)

    for epoch in range(0, cfg.epochs):
        model.train()
        record_list = build_record(cfg.MODEL)
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

            if i % cfg.get("log_interval", 10) == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / record_list["batch_time"].val)
                log_batch(record_list, i, epoch + 1, cfg.epochs, "train", ips)

            tic = time.time()
            pass
