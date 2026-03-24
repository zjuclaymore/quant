"""
全局日志管理模块 (Logger)

提供一致的终端与文件双向输出日志。
"""

import os
import logging
import warnings

warnings.filterwarnings("ignore")

def setup_logger(name: str, out_dir: str = None) -> logging.Logger:
    """
    配置双路输出日志 (Console + File)
    
    参数:
        name: logger 名称
        out_dir: 日志输出目录，如果提供则会额外输出到 backtest_debug.log
        
    返回:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 Handler
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)

        # File handler (if out_dir is provided)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            log_file = os.path.join(out_dir, "backtest_debug.log")
            
            # 使用 w 模式覆盖旧的测试日志，如果想追加可改 a 
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
                )
            )
            logger.addHandler(fh)
            
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    获取现有的 logger 实例，若不存在则创建只输出到终端的 logger
    """
    return logging.getLogger(name)

if __name__ == "__main__":
    # Test
    log = setup_logger("test", "./")
    log.info("Logger initialized.")
    log.debug("This is a debug message.")
