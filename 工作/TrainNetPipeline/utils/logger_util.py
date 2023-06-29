# encoding:utf-8
import logging


def loggers(cfg):
    logs = logging.getLogger("USER")
    logs.setLevel(logging.INFO)
    handler = logging.FileHandler(filename=cfg['log_filename_train_verify'], encoding="UTF-8")
    formatter = logging.Formatter(fmt="%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    if not logs.handlers:
        logs.addHandler(handler)    

    output_config(cfg, logs)

    return logs


def output_config(cfg, logs):
    for key, val in cfg.items():
        logs.info(f'{key}:{val}')
