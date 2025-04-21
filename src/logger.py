"""loggerのユーティリティモジュール"""
import logging


def get_logger_with_default_conf(log_name):
    """デフォルト設定のロガーを取得する"""
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    logger_sh = logging.StreamHandler()
    logger_sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(logger_sh)

    return logger
