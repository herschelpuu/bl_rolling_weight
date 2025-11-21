import io
import sys
import logging
import warnings
from pathlib import Path
from sklearn import set_config
from typing import Optional

# =============================================
# 初始化设置
# =============================================
class EnvironmentManager:
    """处理环境和框架级初始化任务"""

    @staticmethod
    def initialize(log_file: Optional[str] = None) -> None:
        """
        初始化运行环境（包括统一的 logging 配置）。
        如果需要指定日志文件路径，可传入 log_file（字符串或 Path）。
        """
        # 设置标准输出编码为utf-8
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

        # 屏蔽警告
        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore")

        # 配置日志级别为ERROR，减少输出
        logging.getLogger().setLevel(logging.ERROR)
        for logger_name in logging.Logger.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        # 单独设置qlib的日志级别
        qlib_logger = logging.getLogger("qlib")
        qlib_logger.setLevel(logging.ERROR)
        for handler in qlib_logger.handlers[:]:
            qlib_logger.removeHandler(handler)

        # 初始化sklearn配置
        set_config(transform_output="pandas")

        # 默认日志路径（与原来 main 中一致）
        log_path = Path(log_file) if log_file else Path("temp/logs/rolling_weights.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 移除之前可能存在的 handler，避免重复输出
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_path, mode="a", encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.getLogger(__name__).info("Environment initialized, logging -> %s", str(log_path))