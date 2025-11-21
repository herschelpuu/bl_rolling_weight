from utils import *
from pathlib import Path
from config import Config
from typing import List, Optional, Dict, Tuple
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from _single_roll import _single_roll
from utils import _apply_sticky_weights
from _spearman_guard import _apply_spearman_guard


class RollingWeightPipeline:
    """封装并行滚动训练、后处理及落盘逻辑"""

    def __init__(
        self,
        config: Config,
        raw_data: pd.DataFrame,
        returns: pd.DataFrame,
        manual_view_overrides: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Bundle rolling execution inputs and the optional overrides."""
        self.config = config
        self.raw_data = raw_data
        self.returns = returns
        self.manual_view_overrides = manual_view_overrides or {}
        self.logger = logger or logging.getLogger(__name__)

    def execute(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the rolling optimization, apply guards, and return merged outputs."""
        actual_rolling = min(self.config.ROLLING_WINDOW, len(self.returns))
        self.logger.info("开始并行滚动训练，总样本数: %d, 窗口大小(实际): %d", len(self.returns), actual_rolling)
        args_list = [
            (i, self.returns, actual_rolling, self.config, self.raw_data, self.manual_view_overrides)
            for i in range(1, actual_rolling + 1)
        ]
        max_workers = max(1, mp.cpu_count() // 2)
        self.logger.info("使用 %d 个进程进行并行计算", max_workers)

        weight_results: List[pd.DataFrame] = []
        view_results: List[pd.DataFrame] = []
        regime_results: List[pd.DataFrame] = []
        factor_results: List[pd.DataFrame] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_single_roll, arg): arg[0] for arg in args_list}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self.logger.exception("迭代 %d 失败", idx)
                    print(f"迭代 {idx} 失败: {exc}")
                    continue
                if not result:
                    self.logger.info("迭代 %d: 子进程返回空结果", idx)
                    continue
                weight_df, last_date, duration, view_df, regime_df, factor_df = result
                if isinstance(weight_df, pd.DataFrame) and weight_df.empty:
                    self.logger.info("迭代 %d: 返回空权重，跳过", idx)
                    continue
                weight_results.append(weight_df)
                if isinstance(view_df, pd.DataFrame) and not view_df.empty:
                    view_results.append(view_df)
                if isinstance(regime_df, pd.DataFrame) and not regime_df.empty:
                    regime_results.append(regime_df)
                if isinstance(factor_df, pd.DataFrame) and not factor_df.empty:
                    factor_results.append(factor_df)
                self.logger.info("迭代 %d: 耗时 %.2f s, 最后日期 %s", idx, duration, last_date)

        if not weight_results:
            raise RuntimeError("没有可用的迭代结果，无法合并权重")

        combined_weights = pd.concat(weight_results, axis=0).sort_index().fillna(0)
        combined_views = pd.concat(view_results, axis=0).sort_index() if view_results else pd.DataFrame()
        combined_regimes = pd.concat(regime_results, axis=0).sort_index() if regime_results else pd.DataFrame()
        combined_factors = pd.concat(factor_results, axis=0).sort_index() if factor_results else pd.DataFrame()

        combined_weights = _apply_sticky_weights(combined_weights, self.config)
        guard_df = pd.DataFrame()
        factor_frames: Dict[str, pd.DataFrame] = {}
        if not combined_factors.empty:
            factor_dates = pd.to_datetime(combined_factors.index)
            combined_factors.index = factor_dates.normalize()
            feature_names = {col.split("::", 1)[0] for col in combined_factors.columns if "::" in col}
            for feature_name in feature_names:
                feature_cols = [c for c in combined_factors.columns if c.startswith(f"{feature_name}::")]
                if not feature_cols:
                    continue
                feature_df = combined_factors[feature_cols].copy()
                feature_df.columns = [c.split("::", 1)[1] for c in feature_cols]
                factor_frames[feature_name] = feature_df
            debug_dir = Path(self.config.OUTPUT_BASE_PATH) / "guard_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            for feature_name, feature_df in factor_frames.items():
                try:
                    feature_path = debug_dir / f"{feature_name}_frame.csv"
                    feature_df.to_csv(feature_path)
                except Exception as exc:  # pragma: no cover
                    logging.getLogger(__name__).warning("failed to persist %s factor frame: %s", feature_name, exc)
        if not combined_views.empty:
            combined_weights, guard_df = _apply_spearman_guard(combined_weights, combined_views, self.raw_data, self.config, factor_frames)

        return combined_weights, combined_views, combined_regimes, guard_df
