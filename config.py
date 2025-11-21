from pathlib import Path
from typing import Dict, List, Tuple
from pathlib import Path as _Path
import logging
import yaml
import pandas as pd

def update_fund_date(file_path: str, new_date: str) -> bool:
    """Update the third column (date) of a tab-separated instrument file in-place."""
    logger = logging.getLogger(__name__)
    updated_rows = 0
    try:
        # 1. 读取原文件（保留制表符分隔）
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()  # 逐行读取，保留原始换行符

        # 2. 处理每一行：只改第三列，保持制表符分隔
        processed_lines = []
        for line in lines:
            line_stripped = line.strip()  # 去除行首尾空白（避免空行干扰）
            if not line_stripped:  # 跳过空行（若文件存在）
                processed_lines.append(line)
                continue

            # 用制表符精确分割（关键：匹配原文件格式）
            parts = line_stripped.split("\t")
            # 校验列数（确保是3列结构，避免异常行）
            if len(parts) >= 3:
                if parts[2] != new_date:
                    updated_rows += 1
                parts[2] = new_date  # 仅更新第三列（目标日期列）
                # 用制表符重新拼接，完全还原原格式
                processed_line = "\t".join(parts) + "\n"
                processed_lines.append(processed_line)
            else:
                # 若出现异常行（非3列），保留原内容并提示
                processed_lines.append(line)
                logger.warning("跳过异常行（列数不符）：%s", line_stripped)

        # 3. 写回原文件（覆盖更新，保持格式）
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(processed_lines)
        if updated_rows > 0:
            logger.info("已将 %s 中的 %d 行目标日期更新为 %s", file_path, updated_rows, new_date)
        else:
            logger.info("%s 无需更新，第三列已为 %s", file_path, new_date)
        return updated_rows > 0

    except FileNotFoundError:
        logger.warning("update_fund_date: 找不到文件 %s", file_path)
    except Exception as e:
        logger.exception("update_fund_date: 处理 %s 时出错", file_path)
    return False

# =============================================
# 全局配置与参数设置
# =============================================
class Config:
    """配置参数类，集中管理所有可配置参数"""

    def __init__(self) -> None:
        # 数据路径配置
        self.QLIB_DATA_PATH = "/home/pu/.qlib/qlib_data/all_fund_data/"  # qlib数据路径 (Ubuntu path)
        self.OUTPUT_BASE_PATH = Path('temp')  # 输出文件基础路径（Ubuntu 路径）
        self.CLUSTER_MAPPING_FILE = self.OUTPUT_BASE_PATH / "cluster_mapping_selected.csv"
        # 尝试加载同目录下的 config.yaml（优先覆盖）
        yaml_config = {}
        try:
            yaml_path = _Path(__file__).parent / "config.yaml"
            if yaml_path.exists():
                with open(yaml_path, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f) or {}
        except Exception:
            logging.getLogger(__name__).exception("加载 config.yaml 时出错，继续使用内置默认")

        # --- 删除或注释掉：这里不要提前把 yaml 应用到未初始化的实例上 ---
        # for k, v in yaml_config.items():
        #     try:
        #         if hasattr(self, k):
        #             cur = getattr(self, k)
        #             if isinstance(cur, dict) and isinstance(v, dict):
        #                 merged = {**cur, **v}
        #                 setattr(self, k, merged)
        #             else:
        #                 # 如果当前属性是 Path 且 yaml 提供的是字符串，则转换为 Path
        #                 if isinstance(cur, Path) and isinstance(v, str):
        #                     setattr(self, k, Path(v))
        #                 else:
        #                     setattr(self, k, v)
        #         else:
        #             # 新增属性：如果看起来像路径且是字符串，保存为 Path 以保持一致
        #             if isinstance(v, str) and (k.endswith("_PATH") or k.endswith("_DIR") or "PATH" in k or "DIR" in k or "OUTPUT_BASE" in k):
        #                 setattr(self, k, Path(v))
        #             else:
        #                 setattr(self, k, v)
        #     except Exception:
        #         logging.getLogger(__name__).exception("应用 config.yaml 项 %s 失败，跳过", k)

        # 确保 OUTPUT_BASE_PATH 和相关衍生路径为 Path（避免 str / str 使用 '/' 报错）
        try:
            self.OUTPUT_BASE_PATH = Path(self.OUTPUT_BASE_PATH)
        except Exception:
            self.OUTPUT_BASE_PATH = Path(str(self.OUTPUT_BASE_PATH))
        # 更新依赖的衍生路径
        self.CLUSTER_MAPPING_FILE = self.OUTPUT_BASE_PATH / "cluster_mapping_selected.csv"
        self.ATTRIB_CSV = str(self.OUTPUT_BASE_PATH / "weights_attribution.csv")
        self.ATTRIB_TXT = str(self.OUTPUT_BASE_PATH / "weights_attribution.txt")
        # 确保有 MARKET 默认值（yaml 会覆盖该属性）
        self.MARKET = getattr(self, "MARKET", "cluster_mapping_selected")
        # 构造 instruments 文件路径（qlib 期望字符串）
        self.file_path = str(Path(self.QLIB_DATA_PATH) / "instruments" / f"{self.MARKET}.txt")

        # 日期范围配置
        self.start_date = "2025-08-01"
        self.end_date = "2025-11-20"
        self.WARMUP_START_DATE = "2022-12-01"
        self.REPORT_START_DATE = self.start_date
        self.REPORT_END_DATE = self.end_date

        # 窗口向前回溯天数（用于训练窗口的起点计算），可在 Config 中修改
        self.WINDOW_LOOKBACK = 60
        trading_days = pd.bdate_range(self.start_date, self.end_date)
        self.trading_days_count = len(trading_days)
        tmp_range = pd.bdate_range(end=self.start_date, periods=self.WINDOW_LOOKBACK)
        self.start_date_opt = tmp_range[0].strftime("%Y-%m-%d")
        warmup_ts = pd.Timestamp(self.WARMUP_START_DATE) if self.WARMUP_START_DATE else pd.Timestamp(self.start_date_opt)
        self.data_start_ts = min(pd.Timestamp(self.start_date_opt), warmup_ts)
        self.TEST_PERIOD: Tuple[str, str] = (self.data_start_ts.strftime("%Y-%m-%d"), self.end_date)

        self.MARKET = "cluster_mapping_selected"
        update_success = update_fund_date(self.file_path, self.end_date)
        if not update_success:
            logging.getLogger(__name__).warning(
                "instrument 文件 %s 第三列未更新为 %s，请检查分隔符/列结构",
                self.file_path,
                self.end_date,
            )
        self.ROLLING_WINDOW = self.trading_days_count

        # 如果在文件开头成功加载了 yaml_config，则在此刻把 yaml 的项覆盖到实例上
        # 并重新计算依赖的派生路径/日期，确保 yaml 真正生效（便于调试和一致性）
        try:
            if yaml_config:
                for k, v in yaml_config.items():
                    try:
                        if hasattr(self, k):
                            cur = getattr(self, k)
                            # 合并 dict 类型配置
                            if isinstance(cur, dict) and isinstance(v, dict):
                                merged = {**cur, **v}
                                setattr(self, k, merged)
                            else:
                                # 保持 Path 类型属性为 Path
                                if isinstance(cur, Path) and isinstance(v, str):
                                    setattr(self, k, Path(v))
                                else:
                                    setattr(self, k, v)
                        else:
                            # 新增属性：若像路径则转换为 Path
                            if isinstance(v, str) and (k.endswith("_PATH") or k.endswith("_DIR") or "PATH" in k or "DIR" in k or "OUTPUT_BASE" in k):
                                setattr(self, k, Path(v))
                            else:
                                setattr(self, k, v)
                    except Exception:
                        logging.getLogger(__name__).exception("应用 config.yaml 项 %s 失败，跳过", k)

                # 重新保证 OUTPUT_BASE_PATH 为 Path，更新依赖路径
                try:
                    self.OUTPUT_BASE_PATH = Path(self.OUTPUT_BASE_PATH)
                except Exception:
                    self.OUTPUT_BASE_PATH = Path(str(self.OUTPUT_BASE_PATH))
                self.CLUSTER_MAPPING_FILE = self.OUTPUT_BASE_PATH / "cluster_mapping_selected.csv"
                self.ATTRIB_CSV = str(self.OUTPUT_BASE_PATH / "weights_attribution.csv")
                self.ATTRIB_TXT = str(self.OUTPUT_BASE_PATH / "weights_attribution.txt")
                # file_path 依赖 MARKET/start/end
                self.MARKET = getattr(self, "MARKET", "cluster_mapping_selected")
                self.file_path = str(Path(self.QLIB_DATA_PATH) / "instruments" / f"{self.MARKET}.txt")

                # 重新计算与日期相关的派生字段（若 YAML 覆盖了 start/end/WARMUP）
                trading_days = pd.bdate_range(getattr(self, "start_date", self.start_date), getattr(self, "end_date", self.end_date))
                self.trading_days_count = len(trading_days)
                tmp_range = pd.bdate_range(end=getattr(self, "start_date", self.start_date), periods=getattr(self, "WINDOW_LOOKBACK", self.WINDOW_LOOKBACK))
                self.start_date_opt = tmp_range[0].strftime("%Y-%m-%d")
                warmup_ts = pd.Timestamp(getattr(self, "WARMUP_START_DATE", self.WARMUP_START_DATE)) if getattr(self, "WARMUP_START_DATE", None) else pd.Timestamp(self.start_date_opt)
                self.data_start_ts = min(pd.Timestamp(self.start_date_opt), warmup_ts)
                self.TEST_PERIOD = (self.data_start_ts.strftime("%Y-%m-%d"), getattr(self, "end_date", self.end_date))
                self.ROLLING_WINDOW = self.trading_days_count
                logging.getLogger(__name__).info("config.yaml 覆盖项已应用: %s", list(yaml_config.keys()))
        except Exception:
            logging.getLogger(__name__).exception("在应用 config.yaml 覆盖时发生错误，继续使用当前配置")

        # 让基线更中性：保留空列表以触发等权兜底
        self.MANUAL_DROP_CODES: List[str] = [
            "sh517180",
            "sh512190",#浙商之江凤凰ETF
            "sh512750",#嘉实中证锐联基本面ETF
            "sh510010",#交银上证180公司治理ETF
            "sh563330",#华泰柏瑞中证A股ETF
            "sz159691",#工银瑞信中证港股通高股息精选ETF
            "sh517520",#永赢中证沪深港黄金产业股票ETF
            "sz159687",#南方基金南方东英富时亚太低碳精选ETF(QDII)
            "sh510410",#博时上证自然资源ETF
            "sh515100",#景顺长城红利低波动100ETF
            "sh512890",#华泰柏瑞中证红利低波ETF

        ]
        self.BENCHMARK_WEIGHTS: Dict[str, float] = {}
        self.BASELINE_MIX_ALPHA: float = 0.75
        self.BASELINE_ALPHA_WEAK_SCALE: float = 0.55
        self.BASELINE_ALPHA_STRONG_SCALE: float = 1.80
        self.BASELINE_ALPHA_FLOOR: float = 0.25
        self.BASELINE_ALPHA_CAP: float = 0.95
        self.RANK_TOP_PCT: float = 0.25
        self.RANK_POWER: float = 3.5
        self.MIN_WEIGHT_THRESHOLD: float = 0.015
        self.MIN_ACTIVE_COUNT: int = 3
        self.MIN_SIGNAL_STRENGTH: float = 0.0003
        self.STRONG_SIGNAL_STRENGTH: float = 0.0015
        self.MIN_VIEW_SPREAD: float = 0.0015
        self.STRONG_VIEW_SPREAD: float = 0.0060
        self.MAX_WEIGHT_CAP: float = 0.40
        self.MOMENTUM_OVERLAY_ALPHA: float = 0.60
        self.MOMENTUM_OVERLAY_TOP_N: int = 6
        self.MOMENTUM_OVERLAY_MIN_POS: int = 4
        self.MOMENTUM_OVERLAY_CAP: float = 0.35
        self.USE_GUARD_BASELINE: bool = False
        self.SPEARMAN_GUARD_WINDOW: int = 5
        self.SPEARMAN_GUARD_MIN_OBS: int = 3
        self.SPEARMAN_GUARD_THRESHOLD: float = -0.050
        self.SPEARMAN_GUARD_ACTIVE_SCALE: float = 0.15
        self.SPEARMAN_POS_THRESHOLD: float = 0.03
        self.SPEARMAN_POS_SCALE: float = 0.80
        self.SPEARMAN_POS_POWER: float = 1.8
        
        self.GUARD_RANGE_TOP_N: int = 10
        self.GUARD_RANGE_THRESHOLD: float = 1.01  # >1 to effectively disable过热触发
        self.GUARD_RANGE_RATIO: float = 0.95
        self.GUARD_RANGE_BLEND: float = 0.05
        self.GUARD_RANGE_BLEND_MAX: float = 0.15
        self.GUARD_RANGE_CAP: float = 0.80
        self.GUARD_MONEYFLOW_THRESHOLD: float = -0.50
        self.GUARD_MONEYFLOW_NEG_RATIO: float = 0.85
        self.GUARD_MONEYFLOW_SCALE: float = 0.15
        self.GUARD_RANGE_SCALE: float = 0.15

        self.MANUAL_VIEW_OVERRIDES: Dict[str, float] = {}
        self.TRADING_DAYS_PER_YEAR: int = 252
        self.BL_VIEW_ANNUALIZED_RETURN: float = 1.00
        self.BL_VIEW_DAILY_CAP: float = 0.008
        self.STICKY_WEIGHT_ALPHA: float = 0.02

        self.FINAL_MIN_WEIGHT: float = 0.05

        # 加强防御仓位的稳健性，避免单一资产拖累
        self.DEFENSIVE_WEIGHTS: Dict[str, float] = {}
        self.DEFENSIVE_REGIME_THRESHOLD: float = -0.60
        self.DEFENSIVE_REGIME_FLOOR: float = -0.90
        self.DEFENSIVE_BLEND_MIN: float = 0.00
        self.DEFENSIVE_BLEND_MAX: float = 0.10
        self.DEFENSIVE_BLEND: float = 0.05
        self.DEFENSIVE_ALPHA_DAMP: float = 0.00
        self.DEFENSIVE_ALPHA_FLOOR: float = 0.00

        self.GUARD_DEFENSIVE_VIEW_THRESHOLD: float = -0.10
        self.GUARD_DEFENSIVE_VIEW_FLOOR: float = -0.40
        self.GUARD_DEFENSIVE_BLEND_MIN: float = 0.00
        self.GUARD_DEFENSIVE_BLEND_MAX: float = 0.15
        self.GUARD_DEFENSIVE_BLEND: float = 0.08

        # 输出文件名配置
        self.MERGED_CSV = "weights_merged.csv"
        self.FILTERED_CSV = "weights_filtered.csv"
        self.NON_ZERO_CSV = "weights_nonzero.csv"
        self.ATTRIB_CSV = str(self.OUTPUT_BASE_PATH / "weights_attribution.csv")
        self.ATTRIB_TXT = str(self.OUTPUT_BASE_PATH / "weights_attribution.txt")

        # 应用 yaml 配置：若已有属性为 dict，则浅合并；否则直接覆盖/新增属性
        for k, v in yaml_config.items():
            try:
                if hasattr(self, k):
                    cur = getattr(self, k)
                    if isinstance(cur, dict) and isinstance(v, dict):
                        merged = {**cur, **v}
                        setattr(self, k, merged)
                    else:
                        setattr(self, k, v)
                else:
                    setattr(self, k, v)
            except Exception:
                logging.getLogger(__name__).exception("应用 config.yaml 项 %s 失败，跳过", k)

        # 如果 YAML 未显式设置 REPORT_START_DATE/REPORT_END_DATE，则使用 start_date/end_date
        #（确保 downstream 使用 config.REPORT_* 时与 YAML 的 start/end 保持一致）
        try:
            if "REPORT_START_DATE" not in yaml_config:
                self.REPORT_START_DATE = getattr(self, "start_date", self.start_date)
            if "REPORT_END_DATE" not in yaml_config:
                self.REPORT_END_DATE = getattr(self, "end_date", self.end_date)
        except Exception:
            logging.getLogger(__name__).exception("设置 REPORT_START/END 时出错，继续使用当前值")

        # ----------------------------
        # 统一规范化与归一化路径字段（保证 main 中可以直接使用）
        # - OUTPUT_BASE_PATH -> pathlib.Path
        # - QLIB_DATA_PATH -> str (qlib 要求字符串路径)
        # - CLUSTER_MAPPING_FILE -> Path
        # - ATTRIB_CSV / ATTRIB_TXT -> str (仍可作为路径字符串)
        # 其它以 _PATH/_DIR 结尾的字段也尝试转换为 Path
        # ----------------------------
        try:
            # QLIB needs a string path
            if hasattr(self, "QLIB_DATA_PATH"):
                self.QLIB_DATA_PATH = str(Path(self.QLIB_DATA_PATH))

            # Ensure OUTPUT_BASE_PATH is a Path
            if hasattr(self, "OUTPUT_BASE_PATH"):
                self.OUTPUT_BASE_PATH = Path(self.OUTPUT_BASE_PATH)

            # Common derived/related paths
            if hasattr(self, "CLUSTER_MAPPING_FILE"):
                self.CLUSTER_MAPPING_FILE = Path(self.CLUSTER_MAPPING_FILE)
            # ATTRIB paths keep as strings (consistent with existing usage)
            if hasattr(self, "ATTRIB_CSV"):
                self.ATTRIB_CSV = str(Path(self.ATTRIB_CSV))
            if hasattr(self, "ATTRIB_TXT"):
                self.ATTRIB_TXT = str(Path(self.ATTRIB_TXT))

            # Normalize any other yaml-provided *_PATH or *_DIR to Path
            for attr_name in list(yaml_config.keys()):
                if attr_name.endswith("_PATH") or attr_name.endswith("_DIR"):
                    try:
                        val = getattr(self, attr_name, None)
                        if isinstance(val, str):
                            setattr(self, attr_name, Path(val))
                    except Exception:
                        logging.getLogger(__name__).exception("规范化 %s 为 Path 失败", attr_name)
        except Exception:
            logging.getLogger(__name__).exception("路径规范化失败")

