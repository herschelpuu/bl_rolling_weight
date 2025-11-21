from env_manager import EnvironmentManager
from config import Config
from universal_builder import UniverseBuilder
from rolling_w_pipe import RollingWeightPipeline
from utils import _load_cluster_map, _prune_small_weights,_compute_blend_series,\
    parse_manual_view_override_string, _write_csv, _index_to_date
import os, sys, shutil, logging, argparse
from pathlib import Path
from typing import Dict, Iterable, Optional
from datetime import date
import pandas as pd
import numpy as np
import qlib
from qlib.config import REG_CN
from data_loader import DataLoader

# =============================================
# 主函数（已改造：日志落盘 + 屏幕双份输出）
# =============================================
def main(config: Config, options: Optional[Dict] = None) -> pd.DataFrame:
    """
    主函数：协调数据准备、并行计算和结果处理的全过程
    返回: 包含权重信息的合并数据框
    """

    # ======= STEP 1. 兼容旧签名与 options 解析 ===========
    # ****** 1.1 解析 options，优先使用传入 options，其次使用 config 中的默认 ******
    # 兼容旧签名：从 options 取值，优先使用 options，其次使用 config
    options = options or {}
    drop_codes = options.get("drop_codes", None)
    manual_view_overrides = options.get("manual_view_overrides", None)

    # ======= STEP 2. 日志初始化 ===========
    logger = logging.getLogger(__name__)

    # ======= STEP 3. qlib 初始化与运行模式 ===========
    # ****** 3.1 初始化 qlib 数据源，读取环境变量决定因子模式 ******
    qlib.init(provider_uri=config.QLIB_DATA_PATH, region=REG_CN)
    logger.info("qlib 初始化完成，数据目录：%s", config.QLIB_DATA_PATH)
    factor_mode = os.environ.get("BL_FACTOR_MODE", "composite").strip().lower()

    # ======= STEP 4. 从 config 获取可覆盖的默认项 ===========
    # ****** 4.1 把 guard/baseline/regime/列填充值等从 config 拉出 ******
    # 从 config 获取可覆盖的默认项（config.yaml 会被 Config 合并加载）
    guard_defaults = getattr(config, "GUARD_DEFAULTS", {})
    baseline_overlay_default = getattr(config, "BASELINE_DEFENSIVE_OVERLAY_DEFAULT", False)
    baseline_blend_default = getattr(config, "BASELINE_DEFENSIVE_BLEND_DEFAULT", np.nan)
    baseline_severity_default = getattr(config, "BASELINE_DEFENSIVE_SEVERITY_DEFAULT", 0.0)
    regime_signal_default = getattr(config, "REGIME_SIGNAL_DEFAULT", np.nan)
    column_fill_defaults = getattr(config, "COLUMN_FILL_DEFAULTS", {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0, "w": 0})

    # ======= STEP 5. 构建标的池 (Universe) ===========
    # ****** 5.1 调用 UniverseBuilder.build() 并处理异常 ******
    print("开始数据准备...")
    builder = UniverseBuilder(config, drop_codes, logger)
    try:
        stockpool = builder.build()
    except RuntimeError as exc:
        logger.error(str(exc))
        print(f"错误：{exc}")
        return pd.DataFrame()
    print(f"原始标的数量: {len(stockpool)}")

    # ======= STEP 6. 加载行情与特征矩阵 ===========
    # ****** 6.1 使用 DataLoader 加载 raw_data 与 特征 X ******
    data_loader = DataLoader(config, stockpool, logger)
    raw_data, X = data_loader.load()
    print('X.columns:', X.columns.tolist())
    print(f"数据准备完成，有效样本数：{len(X)}行")

    # ======= STEP 7. 执行滚动权重管道 ===========
    # ****** 7.1 调用 RollingWeightPipeline.execute() 并捕获错误 ******
    pipeline = RollingWeightPipeline(config, raw_data, X, manual_view_overrides, logger)
    try:
        combined_weights, combined_views, combined_regimes, guard_df = pipeline.execute()
    except RuntimeError as exc:
        logger.error(str(exc))
        print(f"错误：{exc}")
        return pd.DataFrame()
    
    # ======= STEP 8. 可选的 cluster mapping 与 小权重裁剪 ===========
    # ****** 8.1 加载 cluster_map 并根据 FINAL_MIN_WEIGHT 裁剪小权重 ******
    cluster_map: Dict[str, str] = {}
    cluster_mapping_file = getattr(config, "CLUSTER_MAPPING_FILE", None)
    if cluster_mapping_file:
        cluster_map = _load_cluster_map(cluster_mapping_file)
        if cluster_map:
            logger.info("cluster mapping loaded: %d aliases", len(cluster_map))
    final_min_weight = float(getattr(config, "FINAL_MIN_WEIGHT", 0.0))
    if final_min_weight > 0:
        combined_weights = _prune_small_weights(combined_weights, final_min_weight, cluster_map)
    logger.info("开始合并结果并输出...")
    print("开始合并结果并输出...")

    # ======= STEP 9. 把 raw_data 重塑并标准化日期 ===========
    # ****** 9.1 reset_index 并把 datetime 转为 date（为后续 merge 做准备） ******
    new_df = raw_data.rename_axis(index={'instrument': 'code'}).reset_index()
    new_df['datetime'] = pd.to_datetime(new_df['datetime']).dt.date

    # 使用 helper 统一将 wide 表的索引转为 date（简洁且等价）
    combined_weights = _index_to_date(combined_weights)
    if not combined_views.empty:
        combined_views = _index_to_date(combined_views)
    if not guard_df.empty:
        guard_df = _index_to_date(guard_df)

    # ======= STEP 10. wide -> long 并 merge 到 new_df ===========
    # ****** 10.1 将 weights/views melt 为长表再 merge（按 datetime, code） ******
    combined_weights = combined_weights.copy()
    combined_weights.index.name = "datetime"
    weights_long = combined_weights.reset_index().melt(id_vars="datetime", var_name="code", value_name="w")
    # index 已为 date，保持行为一致，再次规范化（无副作用）
    weights_long["datetime"] = pd.to_datetime(weights_long["datetime"]).dt.date
    new_df = new_df.merge(weights_long, how="left", on=["datetime", "code"])

    if not combined_views.empty:
        combined_views = combined_views.copy()
        combined_views.index.name = "datetime"
        views_long = combined_views.reset_index().melt(id_vars="datetime", var_name="code", value_name="view")
        views_long["datetime"] = pd.to_datetime(views_long["datetime"]).dt.date
        new_df = new_df.merge(views_long, how="left", on=["datetime", "code"])
    else:
        new_df["view"] = np.nan

    # ======= STEP 11. 计算 regime / baseline 并映射到 new_df ===========
    # ****** 11.1 计算 regime 汇总、blend 与 severity，并把结果 map 到 new_df ******
    baseline_overlay_map: Dict[date, bool] = {}
    baseline_blend_map: Dict[date, float] = {}
    baseline_severity_map: Dict[date, float] = {}
    regime_signal_map: Dict[date, float] = {}
    if not combined_regimes.empty:
        # 向量化：wide -> long 然后 merge（避免逐行 apply）
        cr = combined_regimes.copy()
        cr.index.name = "datetime"
        cr_long = cr.reset_index().melt(id_vars="datetime", var_name="code", value_name="view_regime")
        cr_long["datetime"] = pd.to_datetime(cr_long["datetime"]).dt.date
        new_df = new_df.merge(cr_long, how="left", on=["datetime", "code"])
        regime_signal_series = combined_regimes.mean(axis=1, skipna=True)
        regime_signal_series.index = combined_regimes.index
        regime_signal_map = regime_signal_series.to_dict()
    else:
        new_df['view_regime'] = np.nan

    # ======= STEP 12. Guard 映射与默认填充 ===========
    # ****** 12.1 用 guard_df 映射若存在的列，否则初始化为 NaN ******
    # 向量化 guard 映射：逐列 melt -> merge；默认值来自 config.GUARD_DEFAULTS
    guard_mapping = [
        ("guard_avg", "spearman_guard_avg"),
        ("guard_trigger", "spearman_guard_trigger"),
        ("guard_scale", "spearman_guard_scale"),
        ("spearman_realized", "spearman_realized"),
        ("pos_boost", "spearman_pos_boost"),
        ("pos_scale", "spearman_pos_scale"),
        ("defensive_overlay", "defensive_overlay"),
        ("defensive_blend", "defensive_blend"),
        ("defensive_severity", "defensive_severity"),
        ("range_overheat", "range_overheat"),
        ("range_ratio_top", "range_ratio_top"),
        ("money_outflow", "money_outflow"),
        ("money_avg_top", "money_avg_top"),
        ("money_neg_ratio_top", "money_neg_ratio_top"),
        ("range_cap_applied", "range_cap_applied"),
    ]
    if not guard_df.empty:
        g = guard_df.copy()
        g.index.name = "datetime"
        for src_col, dst_col in guard_mapping:
            if src_col in g.columns:
                tmp = g[[src_col]].reset_index().rename(columns={src_col: dst_col})
                tmp["datetime"] = pd.to_datetime(tmp["datetime"]).dt.date
                # 如果 tmp 包含 code 列，按 datetime+code 合并；否则只按 datetime 合并（广播到所有 code）
                if "code" in tmp.columns:
                    new_df = new_df.merge(tmp, how="left", on=["datetime", "code"])
                else:
                    new_df = new_df.merge(tmp, how="left", on=["datetime"])
            else:
                new_df[dst_col] = np.nan
    else:
        for _, dst_col in guard_mapping:
            new_df[dst_col] = np.nan

    # 用 config.GUARD_DEFAULTS 填充默认值（保持 None -> 不填）
    for _, dst_col in guard_mapping:
        default = guard_defaults.get(dst_col, None)
        if default is None:
            continue
        if dst_col in new_df.columns:
            new_df[dst_col] = new_df[dst_col].fillna(default)

    # ======= STEP 13. 把 regime/baseline 字段映射到 new_df ===========
    # ****** 13.1 将 regime_signal_mean 与 baseline_defensive_* 字段写入 new_df（含默认） ******
    # regime / baseline 字段使用 config 中的默认覆盖
    if regime_signal_map:
        new_df['regime_signal_mean'] = new_df['datetime'].map(regime_signal_map)
    else:
        new_df['regime_signal_mean'] = regime_signal_default
    if baseline_overlay_map:
        new_df['baseline_defensive_overlay'] = new_df['datetime'].map(baseline_overlay_map).fillna(baseline_overlay_default)
    else:
        new_df['baseline_defensive_overlay'] = baseline_overlay_default
    if baseline_blend_map:
        new_df['baseline_defensive_blend'] = new_df['datetime'].map(baseline_blend_map)
    else:
        new_df['baseline_defensive_blend'] = baseline_blend_default
    if baseline_severity_map:
        new_df['baseline_defensive_severity'] = new_df['datetime'].map(baseline_severity_map).fillna(baseline_severity_default)
    else:
        new_df['baseline_defensive_severity'] = baseline_severity_default
    new_df = new_df.set_index(['code', 'datetime'])
    # 计算 next_close（close 已在之前统一去 $）
    if 'next_close' not in new_df.columns:
        new_df['next_close'] = new_df.groupby('code')['close'].shift(-1)
    new_df = new_df.sort_index(
    level=['datetime', 'code'],  # 先按datetime排序，再按code排序
    ascending=[True, False]      # datetime升序，code降序
    )

    # ======= STEP 14. 列填充值（来自 config） ===========
    # ****** 14.1 使用 config.COLUMN_FILL_DEFAULTS 填充缺失列 ******
    # （列默认填充值已在 DataLoader 中完成；这里仅做额外保险性填充）
    safe_fill = {k: v for k, v in getattr(config, "COLUMN_FILL_DEFAULTS", {}).items() if k in new_df.columns}
    if safe_fill:
        new_df = new_df.fillna(value=safe_fill)

    # ======= STEP 15. 写入输出文件（MERGED/FILTERED/NON_ZERO） ===========
    # ****** 15.1 确保 OUTPUT_BASE_PATH 作为 Path 使用并写入文件 ******
    # config.OUTPUT_BASE_PATH 已为 Path，直接使用
    output_path = config.OUTPUT_BASE_PATH / config.MERGED_CSV
    _write_csv(output_path, new_df, logger)
 
    # 过滤、非零文件同理
    new_df_filtered = new_df.reset_index()
    # 使用更简洁的 pd.to_datetime(errors='coerce')，与原来效果等价
    new_df_filtered['datetime'] = pd.to_datetime(new_df_filtered['datetime'], errors='coerce')
    report_start = getattr(config, "REPORT_START_DATE", config.start_date)
    report_end = getattr(config, "REPORT_END_DATE", config.end_date)
    report_start_dt = pd.to_datetime(report_start, errors='coerce')
    report_end_dt = pd.to_datetime(report_end, errors='coerce')
    if pd.isna(report_start_dt):
        report_start_dt = pd.to_datetime(config.start_date, errors='coerce')
    if pd.isna(report_end_dt):
        report_end_dt = pd.to_datetime(config.end_date, errors='coerce')
    new_df_filtered = new_df_filtered[
        (new_df_filtered['datetime'] >= report_start_dt) &
        (new_df_filtered['datetime'] <= report_end_dt)
    ].copy()
    filtered_path = config.OUTPUT_BASE_PATH / config.FILTERED_CSV
    _write_csv(filtered_path, new_df_filtered, logger)
 
     # 保存单因子专用文件，避免回测阶段引用旧数据
    if factor_mode:
        suffix = factor_mode.replace(" ", "_")
        factor_filtered_name = f"weights_filtered_{suffix}.csv"
        factor_filtered_path = config.OUTPUT_BASE_PATH / factor_filtered_name
        shutil.copy(filtered_path, factor_filtered_path)
        logger.info("因子模式 %s 文件已保存至: %s", factor_mode, factor_filtered_path)
        print(f"因子模式 {factor_mode} 文件已保存至: {factor_filtered_path}")
 
    non_zero_df = new_df_filtered[new_df_filtered['w'] > 0]
    non_zero_path = config.OUTPUT_BASE_PATH / config.NON_ZERO_CSV
    _write_csv(non_zero_path, non_zero_df, logger)

    logger.info("最终数据维度: %s", str(new_df.shape))
    print(f"最终数据维度: {new_df.shape}")
    return new_df

# =============================================
# 程序入口
# =============================================
if __name__ == "__main__":

    EnvironmentManager.initialize()     # 初始化环境 设置日志等

    config = Config()   # 加载配置

    parser = argparse.ArgumentParser(description="Generate rolling Black-Litterman weights")
    parser.add_argument(
        "--drop", metavar="CODES", type=str, default="",
        help="Comma-separated list of ETF codes to exclude from the MARKET universe"
    )
    parser.add_argument(
        "--view-override", metavar="CODE=VALUE,...", type=str, default="",
        help="Comma-separated overrides for analyst views, e.g. SH513880=0.02"
    )
    args = parser.parse_args()
    cli_drop_codes = [c.strip() for c in args.drop.split(',') if c.strip()]
    cli_view_overrides = parse_manual_view_override_string(args.view_override)

    # 运行主函数
    combined_drop = list({*config.MANUAL_DROP_CODES, *cli_drop_codes})
    combined_view_overrides = dict(config.MANUAL_VIEW_OVERRIDES)
    combined_view_overrides.update(cli_view_overrides)

    options = {"drop_codes": combined_drop, "manual_view_overrides": combined_view_overrides}
    result_df = main(config, options=options)

    print("程序执行完成")