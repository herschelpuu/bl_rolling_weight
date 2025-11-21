from typing import Iterable, List, Optional, Tuple
import logging
import pandas as pd
from qlib.data import D
from skfolio.preprocessing import prices_to_returns
from config import Config

class DataLoader:
    """负责从 qlib 读取行情数据，并转换为收益率矩阵"""

    def __init__(
        self,
        config: Config,
        stockpool: Iterable[str],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Record the basic data loading parameters."""
        self.config = config
        self.stockpool = list(stockpool)
        self.logger = logger or logging.getLogger(__name__)

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Pull raw price panels and convert them into a daily returns matrix."""
        if not self.stockpool:
            raise ValueError("stockpool 为空，无法加载数据")
        raw_data = D.features(
            self.stockpool,
            fields=["$open", "$high", "$low", "$close", "$volume"],
            start_time=self.config.TEST_PERIOD[0],
            end_time=self.config.TEST_PERIOD[1],
            freq="day",
        )
        # 1) 统一去掉列名中的 $（保持第一层 field 名称去 $）
        try:
            cols = raw_data.columns
            if isinstance(cols, pd.MultiIndex):
                new_tuples = []
                for lvl0, lvl1 in cols:
                    new_lvl0 = lvl0.replace("$", "") if isinstance(lvl0, str) else lvl0
                    new_tuples.append((new_lvl0, lvl1))
                raw_data.columns = pd.MultiIndex.from_tuples(new_tuples)
            else:
                raw_data.columns = [c.replace("$", "") if isinstance(c, str) else c for c in cols]
        except Exception:
            self.logger.exception("尝试去掉 raw_data 列名中的 $ 时出错，继续")

        # 2) 使用 config 中的 COLUMN_FILL_DEFAULTS 对各字段做默认填充（按 field 层级）
        try:
            fill_defaults = getattr(self.config, "COLUMN_FILL_DEFAULTS", {})
            if fill_defaults:
                # 对 MultiIndex 的情况，按第一层 field 名称填充每个 field 对应的子表
                if isinstance(raw_data.columns, pd.MultiIndex):
                    fields = raw_data.columns.get_level_values(0).unique()
                    for field, default in fill_defaults.items():
                        if field in fields:
                            raw_data[field] = raw_data[field].fillna(default)
                else:
                    for field, default in fill_defaults.items():
                        if field in raw_data.columns:
                            raw_data[field] = raw_data[field].fillna(default)
                self.logger.info("raw_data 已按 COLUMN_FILL_DEFAULTS 填充缺失值")
        except Exception:
            self.logger.exception("尝试按 COLUMN_FILL_DEFAULTS 填充 raw_data 时出错")

        # 3) 生成 close 面板并计算 returns
        try:
            close_data = raw_data['close'].unstack(level='instrument').rename_axis(index='Date')
            returns = prices_to_returns(close_data)
            self.logger.info("数据准备完成，有效样本数：%d 行", len(returns))
            return raw_data, returns
        except Exception:
            self.logger.exception("从 raw_data 生成 close/returns 时出错")
            raise