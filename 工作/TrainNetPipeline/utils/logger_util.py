# encoding:utf-8
import os
import logging


def loggers(save_dir, filename, cfg):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logs = logging.getLogger("USER")
    logs.setLevel(logging.INFO)
    handler = logging.FileHandler(filename=os.path.join(save_dir, filename), encoding="UTF-8")
    formatter = logging.Formatter(fmt="%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    if not logs.handlers:
        logs.addHandler(handler)    

    output_config(cfg, logs)

    return logs


def output_config(cfg, logs):
    for key, val in cfg.items():
        logs.info(f'{key}:{val}')
