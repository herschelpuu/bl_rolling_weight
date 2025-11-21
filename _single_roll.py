from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np
import time
import logging
from config import Config
from pathlib import Path
# 投资组合优化库
from sklearn import set_config
from sklearn.pipeline import Pipeline
from skfolio import RiskMeasure
from skfolio.moments import DenoiseCovariance,ShrunkMu
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior,BlackLitterman
from skfolio.preprocessing import prices_to_returns

from generate_fac_view import generate_gap_factor_views
from utils import _parse_view_strings, _cap_weights
from utils import _build_baseline_series, _build_weight_series


# =============================================
# 并行计算相关函数
# =============================================
def _single_roll(args: Tuple[int, pd.DataFrame, int, Config, pd.DataFrame, Optional[Dict[str, float]]]) -> Tuple[pd.DataFrame, pd.Timestamp, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    单个滚动窗口的训练函数，供并行进程调用

    参数:
        args: 包含以下元素的元组
            i: 当前迭代索引
            X_total: 完整的特征数据
            new_i: 滚动窗口大小

    返回:
        Tuple: 包含权重数据框、最后日期和计算耗时的元组
    """
    import os, pathlib, pandas as pd, numpy as np

    # 子进程专用日志句柄（行缓冲，实时可见）
    LOG_FILE = pathlib.Path('temp/roll_subproc.log')
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    sub_logger = open(LOG_FILE, "a", buffering=1, encoding="utf-8")

    def log_sub(msg: str) -> None:
        sub_logger.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | {msg}\n")
    i, X_total, new_i, config, raw_data, manual_view_overrides = args
    t0 = time.perf_counter()  # 计时开始

    # --------------------------------------------------
    # 滑动窗口：只取计算日（含）往前 x 个交易日
    # --------------------------------------------------
    current_end_idx = len(X_total) - new_i + i  # 当前计算日对应的位置（左闭右开）
    window_start_idx = max(0, current_end_idx - config.WINDOW_LOOKBACK)  # 往前推 x 根
    X_train = X_total.iloc[window_start_idx:current_end_idx].copy()
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    zero_cols = X_train.columns[(X_train.abs().sum(axis=0) == 0.0)]
    if len(zero_cols) > 0:
        logging.info("%s zero-variance assets dropped: %s", pd.Timestamp.now(), zero_cols.tolist())
        X_train = X_train.drop(columns=zero_cols)
    # 如果训练集为空，记录并提前返回（避免后续模型在空数据上报错）
    if X_train.shape[0] == 0:
        sub_logger.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | WARNING: iteration {i} has empty training set (window_start={window_start_idx}, end={current_end_idx})\n")
        # 尝试返回回退等权重（当可能时），以避免主进程因无结果而中止
        try:
            # 尝试基于 X_total 的列构造等权重
            assets = list(X_total.columns)
            if len(assets) == 0:
                sub_logger.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | ERROR: no assets available to build fallback weights\n")
                sub_logger.close()
                return pd.DataFrame(), None, 0.0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            w = np.full(len(assets), 1.0 / len(assets))
            # 尝试使用一个合理的 last_date
            try:
                last_date = X_total.index[max(0, current_end_idx - 1)]
            except Exception:
                last_date = pd.Timestamp.now()
            weight_df = pd.DataFrame([dict(zip(assets, w))], index=[last_date])
            sub_logger.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | INFO: returned fallback equal-weight for empty training set\n")
            sub_logger.close()
            return weight_df, last_date, 0.0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        except Exception:
            sub_logger.close()
            return pd.DataFrame(), None, 0.0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # --------------------------------------------------

    # 生成分析师观点（此处可根据实际需求修改观点生成逻辑）
    # 示例：简单生成每个资产的中性观点
    # 生成因子观点：基于指定时间点的涨跌幅
    # ---------------------- 修改后的数据处理逻辑 ----------------------
    # 计算当前窗口在Y_total中的结束索引（当前窗口最后一个位置，即上一个交易日）
    current_window_end_idx = len(X_total) - new_i + i - 1  # 减1是因为iloc切片左闭右开

    # 使用封装的函数生成因子观点，可以选择T+1日(1)、T日(0)或T-1日(-1)
    analyst_views = generate_gap_factor_views(
        X_total,
        raw_data,
        current_window_end_idx,
        time_offset=1,
        manual_view_overrides=manual_view_overrides,
        config=config,
    )
    view_dict = _parse_view_strings(analyst_views)
    # 2. Black-Litterman 后验估计器
    bl = BlackLitterman(
        views=analyst_views,          # 你的观点 Series/DataFrame
    )

    # 3. 优化器
    optimizer = MeanRisk(
        #risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        prior_estimator=bl,           # 把“后验”当最终先验喂给优化器
        #prior_estimator=emp_prior,
        portfolio_params=dict(name="Black & Litterman Model")
        )
    
    model = Pipeline([
        ("optimization", optimizer)
    ])
    model.fit(X_train)

    # 提取权重
    opt = model.named_steps["optimization"]
    weights = pd.DataFrame({
        "asset": opt.feature_names_in_,
        "MeanRisk": opt.weights_
    })
    weights = weights[weights["MeanRisk"] != 0.0]  # 过滤掉权重为0的资产

    # MeanRisk 求解后、过滤前
    # 先拿日期
    last_date = X_train.index[-1]

    raw_w = opt.weights_
    # logging.info('%s raw weights: %s', last_date, dict(zip(opt.feature_names_in_, raw_w)))
    # print(f'{last_date}  raw weights:', dict(zip(opt.feature_names_in_, raw_w)))

    # ===== 拆盒落盘（仅一行）=====
    # ===== 拆盒 + 落盘（仅 5 行）=====
    μ_bl = X_train.mean()
    Σ = X_train.cov()
    if not np.allclose(Σ.values, Σ.values.T, atol=1e-12):
        logging.warning("covariance matrix asymmetric; enforcing symmetry")
        Σ = (Σ + Σ.T) / 2
    w = opt.weights_
    sharpe_1d = μ_bl / np.sqrt(np.diag(Σ))
    port_sharpe = (w @ μ_bl) / np.sqrt(w @ Σ @ w)

    # 落盘 CSV（追加写）
    row = pd.Series({
        'datetime': last_date,
        **{f'μBL_{c}': μ_bl[c] for c in μ_bl.index},
        **{f'σ_{c}': np.sqrt(Σ.loc[c, c]) for c in Σ.index},
        **{f'Sharpe_{c}': sharpe_1d[c] for c in μ_bl.index},
        **{f'weight_{c}': w[i] for i, c in enumerate(μ_bl.index)},
        'port_sharpe': port_sharpe
    }, name=last_date)
    fname = config.ATTRIB_CSV
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    row.to_frame().T.to_csv(fname, mode='a', header=not os.path.exists(fname), index=False)

    # 实时写子进程日志
    log_sub(f"==== {last_date} 拆盒 ====")
    log_sub("μ_BL:\n" + μ_bl.to_string())
    log_sub("最终权重:\n" + pd.Series(w, index=μ_bl.index, name='weight').round(6).to_string())
    log_sub(f"组合夏普 = {port_sharpe:.6f}")
    # ===== 落盘结束 =====

    weight_df, view_df, regime_df, blend_stats = _blend_weight_outputs(
        X_total=X_total,
        weights=weights,
        view_dict=view_dict,
        config=config,
        last_date=last_date,
    )
    alpha = blend_stats["alpha"]
    combined_strength = blend_stats["combined_strength"]
    signal_strength = blend_stats["signal_strength"]
    view_spread = blend_stats["view_spread"]
    baseline_def_blend = blend_stats["baseline_def_blend"]
    baseline_def_severity = blend_stats["baseline_def_severity"]

    duration = time.perf_counter() - t0  # 计算耗时
    ##########################################################
    corr = X_train.corrwith((X_train @ w), axis=0)  # 各资产与组合的相关系数
    log_sub("与组合相关系数:\n" + corr.to_string())
    # invert Σ with jitter fallback in case Σ is singular or near-singular
    try:
        Σ_inv = np.linalg.inv(Σ)
    except np.linalg.LinAlgError:
        eps = 1e-8
        max_tries = 6
        for i in range(max_tries):
            try:
                Σ_inv = np.linalg.inv(Σ + np.eye(Σ.shape[0]) * eps)
                sub_logger.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | INFO: Σ inversion succeeded with jitter eps={eps}\n")
                break
            except np.linalg.LinAlgError:
                eps *= 10
        else:
            raise
    component = Σ_inv @ w  # 协方差逆分量
    rank = pd.Series(component, index=μ_bl.index).sort_values(ascending=False)
    log_sub("【分量排名】\n" + rank.head(5).to_string())
    #########################################################
    sub_logger.close()  # ← 强制刷缓冲 + 关闭
    factor_records: Dict[str, float] = {
        "alpha_mix": float(alpha),
        "combined_strength": float(combined_strength),
        "signal_strength": float(signal_strength),
        "view_spread": float(view_spread),
        "baseline_def_blend": float(baseline_def_blend),
        "baseline_def_severity": float(baseline_def_severity),
    }
    factor_meta = getattr(generate_gap_factor_views, "_last_factors", {})
    if isinstance(factor_meta, dict):
        for feature_name in ("range_score", "money_flow"):
            series = factor_meta.get(feature_name)
            if not isinstance(series, pd.Series):
                continue
            for code, value in series.items():
                key = f"{feature_name}::{code}"
                factor_records[key] = float(value) if np.isfinite(value) else np.nan
    factor_df = pd.DataFrame([factor_records], index=[last_date])

    return weight_df, last_date, duration, view_df, regime_df, factor_df


def _blend_weight_outputs(
    X_total: pd.DataFrame,
    weights: pd.DataFrame,
    view_dict: Dict[str, float],
    config: Config,
    last_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Combine optimizer weights with ranking, baselines, overlays, and diagnostics."""
    all_assets = list(X_total.columns)
    final_series = weights.set_index("asset")["MeanRisk"].astype(float)
    final_series = final_series.reindex(all_assets).fillna(0.0)

    view_series = pd.Series(view_dict, dtype=float).reindex(all_assets)
    raw_gap_series = pd.Series(np.nan, index=all_assets, dtype=float)
    scaled_view_series = pd.Series(np.nan, index=all_assets, dtype=float)

    regime_meta = getattr(generate_gap_factor_views, "_last_regime", {})
    regime_series_raw = regime_meta.get("regime_score") if isinstance(regime_meta, dict) else None
    if isinstance(regime_series_raw, pd.Series):
        regime_series = regime_series_raw.reindex(all_assets)
    else:
        regime_series = pd.Series(np.nan, index=all_assets, dtype=float)

    if isinstance(regime_meta, dict):
        raw_gap_candidate = regime_meta.get("gap_values")
        if isinstance(raw_gap_candidate, pd.Series):
            raw_gap_series = raw_gap_candidate.reindex(all_assets).astype(float)
        scaled_view_candidate = regime_meta.get("scaled_views")
        if isinstance(scaled_view_candidate, pd.Series):
            scaled_view_series = scaled_view_candidate.reindex(all_assets).astype(float)
        momentum_candidate = regime_meta.get("ema_mom")
        if isinstance(momentum_candidate, pd.Series):
            momentum_values = momentum_candidate.reindex(all_assets).astype(float).fillna(0.0)
        else:
            momentum_values = pd.Series(0.0, index=all_assets, dtype=float)
    else:
        momentum_values = pd.Series(0.0, index=all_assets, dtype=float)

    regime_signal = np.nan
    regime_valid = regime_series.dropna()
    if not regime_valid.empty:
        regime_signal = float(regime_valid.mean())

    rank_pct = view_series.rank(ascending=False, method="average", pct=True).fillna(1.0)
    min_active = int(getattr(config, "MIN_ACTIVE_COUNT", 5))
    rank_cutoff = float(getattr(config, "RANK_TOP_PCT", 0.45))
    rank_mask = rank_pct <= rank_cutoff
    if rank_mask.sum() < max(1, min_active):
        top_indices = rank_pct.nsmallest(max(1, min_active)).index
        rank_mask.loc[top_indices] = True
    rank_power = getattr(config, "RANK_POWER", 2.0)
    rank_weight = np.power(np.clip(1.0 - rank_pct, 0.0, 1.0), rank_power)
    rank_weight = rank_weight.where(rank_mask, 0.0)

    ranked_series = final_series * rank_weight
    ranked_sum = ranked_series.sum()
    if ranked_sum <= 0:
        ranked_series = final_series.copy()
        ranked_sum = ranked_series.sum()
    if ranked_sum <= 0:
        ranked_series = pd.Series(1.0, index=all_assets, dtype=float)
        ranked_sum = ranked_series.sum()
    ranked_series = ranked_series / ranked_sum

    baseline_series = _build_baseline_series(all_assets, config)
    defensive_series = _build_weight_series(all_assets, getattr(config, "DEFENSIVE_WEIGHTS", {}))
    defensive_threshold = float(getattr(config, "DEFENSIVE_REGIME_THRESHOLD", np.nan))
    defensive_floor = float(getattr(config, "DEFENSIVE_REGIME_FLOOR", defensive_threshold - 0.25))
    defensive_blend_min = float(np.clip(getattr(config, "DEFENSIVE_BLEND_MIN", getattr(config, "DEFENSIVE_BLEND", 0.35)), 0.0, 1.0))
    defensive_blend_max = float(np.clip(getattr(config, "DEFENSIVE_BLEND_MAX", getattr(config, "DEFENSIVE_BLEND", 0.65)), 0.0, 1.0))
    if defensive_blend_max < defensive_blend_min:
        defensive_blend_min, defensive_blend_max = defensive_blend_max, defensive_blend_min

    baseline_def_severity = 0.0
    baseline_def_blend = 0.0
    if (
        defensive_series.sum() > 0
        and np.isfinite(defensive_threshold)
        and np.isfinite(regime_signal)
    ):
        baseline_def_severity = _compute_regime_severity(regime_signal, defensive_threshold, defensive_floor)
        if baseline_def_severity > 0.0:
            baseline_def_blend = _compute_blend_value(
                regime_signal,
                defensive_threshold,
                defensive_floor,
                defensive_blend_min,
                defensive_blend_max,
            )
            if baseline_def_blend > 0.0:
                mixed_base = baseline_series * (1.0 - baseline_def_blend) + defensive_series * baseline_def_blend
                mixed_base = mixed_base.clip(lower=0.0)
                mixed_sum = mixed_base.sum()
                if mixed_sum > 0:
                    baseline_series = mixed_base / mixed_sum

    if isinstance(regime_meta, dict):
        regime_meta["regime_signal_mean"] = regime_signal
        regime_meta["baseline_defensive_blend"] = baseline_def_blend
        regime_meta["baseline_defensive_severity"] = baseline_def_severity

    base_alpha = float(getattr(config, "BASELINE_MIX_ALPHA", 0.35))
    weak_scale = float(getattr(config, "BASELINE_ALPHA_WEAK_SCALE", 0.3))
    strong_scale = float(getattr(config, "BASELINE_ALPHA_STRONG_SCALE", 1.0))
    min_strength = float(getattr(config, "MIN_SIGNAL_STRENGTH", 0.18))
    strong_strength = float(getattr(config, "STRONG_SIGNAL_STRENGTH", 0.4))
    min_view_spread = float(getattr(config, "MIN_VIEW_SPREAD", 0.0))
    strong_view_spread = float(getattr(config, "STRONG_VIEW_SPREAD", max(min_view_spread, 1.0)))
    if strong_view_spread <= min_view_spread:
        strong_view_spread = min_view_spread + 0.1
    alpha_floor_cfg = float(getattr(config, "BASELINE_ALPHA_FLOOR", base_alpha * weak_scale))
    alpha_cap_cfg = float(getattr(config, "BASELINE_ALPHA_CAP", base_alpha * strong_scale))
    alpha_low = max(0.0, min(1.0, max(alpha_floor_cfg, base_alpha * weak_scale)))
    alpha_high = max(alpha_low, min(1.0, min(alpha_cap_cfg, base_alpha * strong_scale)))

    active_count = int(rank_mask.sum())
    valid_views = view_series.dropna()
    signal_strength = float(valid_views.abs().mean()) if not valid_views.empty else 0.0
    view_spread = float(valid_views.max() - valid_views.min()) if not valid_views.empty else 0.0

    signal_norm = 0.0
    spread_norm = 0.0
    if active_count >= max(1, min_active):
        if signal_strength > min_strength:
            strength_denom = max(1e-6, strong_strength - min_strength)
            signal_norm = float(np.clip((signal_strength - min_strength) / strength_denom, 0.0, 1.0))
        if view_spread > min_view_spread:
            spread_denom = max(1e-6, strong_view_spread - min_view_spread)
            spread_norm = float(np.clip((view_spread - min_view_spread) / spread_denom, 0.0, 1.0))
    combined_strength = min(signal_norm, spread_norm)

    if combined_strength <= 0.0:
        alpha = alpha_low
    else:
        alpha = alpha_low + (alpha_high - alpha_low) * combined_strength
    defensive_alpha_damp = float(np.clip(getattr(config, "DEFENSIVE_ALPHA_DAMP", 0.0), 0.0, 1.0))
    defensive_alpha_floor_cfg = float(getattr(config, "DEFENSIVE_ALPHA_FLOOR", alpha_low))
    if baseline_def_severity > 0.0 and defensive_alpha_damp > 0.0:
        damp_factor = max(0.0, 1.0 - baseline_def_severity * defensive_alpha_damp)
        alpha *= damp_factor
    alpha_floor_effective = max(0.0, max(alpha_floor_cfg, defensive_alpha_floor_cfg))
    alpha_cap_effective = min(1.0, alpha_cap_cfg)
    alpha = float(np.clip(alpha, alpha_floor_effective, alpha_cap_effective))

    blended_series = baseline_series * (1.0 - alpha) + ranked_series * alpha
    blended_series = blended_series.clip(lower=0.0)
    blended_sum = blended_series.sum()
    if blended_sum > 0:
        blended_series = blended_series / blended_sum
    else:
        blended_series = baseline_series.copy()

    blended_series = blended_series.where(rank_mask | (baseline_series > 0), 0.0)
    if blended_series.sum() == 0:
        blended_series = baseline_series.copy()

    min_threshold = float(getattr(config, "MIN_WEIGHT_THRESHOLD", 0.005))
    blended_series = blended_series.where(blended_series >= min_threshold, 0.0)
    if blended_series.sum() == 0:
        blended_series = baseline_series.copy()
    blended_series = blended_series / blended_series.sum()

    mom_alpha = float(np.clip(getattr(config, "MOMENTUM_OVERLAY_ALPHA", 0.0), 0.0, 1.0))
    if mom_alpha > 0.0 and isinstance(momentum_values, pd.Series):
        mom_series = momentum_values.reindex(all_assets).astype(float)
        mom_series = mom_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mom_series = mom_series.clip(lower=0.0)
        top_n = max(1, int(getattr(config, "MOMENTUM_OVERLAY_TOP_N", 5)))
        min_pos = max(1, int(getattr(config, "MOMENTUM_OVERLAY_MIN_POS", 1)))
        mom_sorted = mom_series.sort_values(ascending=False)
        mom_candidates = mom_sorted.head(top_n)
        mom_candidates = mom_candidates[mom_candidates > 0]
        if len(mom_candidates) >= min_pos and mom_candidates.sum() > 0:
            mom_weights = pd.Series(0.0, index=all_assets, dtype=float)
            mom_weights.loc[mom_candidates.index] = mom_candidates
            mom_total = mom_weights.sum()
            if mom_total > 0:
                mom_weights = mom_weights / mom_total
            mom_cap = float(getattr(config, "MOMENTUM_OVERLAY_CAP", np.nan))
            if np.isfinite(mom_cap) and 0.0 < mom_cap < 1.0:
                mom_weights = _cap_weights(mom_weights, mom_cap)
            mom_total = mom_weights.sum()
            if mom_total > 0:
                mom_weights = mom_weights / mom_total
            blended_series = blended_series * (1.0 - mom_alpha) + mom_weights * mom_alpha
            blended_series = blended_series.clip(lower=0.0)
            total = blended_series.sum()
            if total > 0:
                blended_series = blended_series / total

    snapshot_df = pd.DataFrame({
        "code": all_assets,
        "gap_value": raw_gap_series,
        "scaled_view": scaled_view_series,
        "view_value": view_series.astype(float),
        "rank_pct": rank_pct.astype(float),
        "rank_weight": rank_weight.astype(float),
        "rank_mask": rank_mask.astype(int),
        "baseline_weight": baseline_series.reindex(all_assets).astype(float),
        "ranked_weight": ranked_series.reindex(all_assets).astype(float),
        "blended_weight": blended_series.reindex(all_assets).astype(float),
    }).set_index("code")
    try:
        signal_debug_dir = Path(config.OUTPUT_BASE_PATH) / "signal_debug"
        signal_debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = signal_debug_dir / f"{last_date.strftime('%Y%m%d')}_signals.csv"
        snapshot_df.to_csv(debug_path)
    except Exception as exc:
        logging.getLogger(__name__).warning("failed to persist signal snapshot for %s: %s", last_date, exc)

    max_cap = float(getattr(config, "MAX_WEIGHT_CAP", 1.0))
    if 0.0 < max_cap < 1.0:
        blended_series = _cap_weights(blended_series, max_cap)

    weight_df = pd.DataFrame([blended_series.to_dict()], index=[last_date])

    view_row = {code: view_dict.get(code, np.nan) for code in X_total.columns}
    view_df = pd.DataFrame([view_row], index=[last_date])

    if isinstance(regime_series, pd.Series):
        regime_aligned = regime_series.reindex(X_total.columns)
    else:
        regime_aligned = pd.Series(np.nan, index=X_total.columns, dtype=float)
    if isinstance(regime_meta, dict):
        regime_meta['regime_signal_mean'] = regime_signal
    regime_row = regime_aligned.to_dict()
    regime_df = pd.DataFrame([regime_row], index=[last_date])

    meta = {
        "alpha": float(alpha),
        "combined_strength": float(combined_strength),
        "signal_strength": float(signal_strength),
        "view_spread": float(view_spread),
        "baseline_def_blend": float(baseline_def_blend),
        "baseline_def_severity": float(baseline_def_severity),
    }

    return weight_df, view_df, regime_df, meta