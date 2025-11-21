from config import Config
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from utils import _normalize_code, _code_aliases

def _apply_manual_view_overrides(series: pd.Series, overrides: Optional[Dict[str, float]]) -> pd.Series:
    """根据手工指定的观点覆盖Series中的对应资产"""
    if not overrides:
        return series
    alias_map: Dict[str, float] = {}
    for code, value in overrides.items():
        try:
            fv = float(value)
        except Exception:
            continue
        for alias in _code_aliases(code):
            alias_map[alias] = fv
    adjusted = series.copy()
    for idx in adjusted.index:
        norm_idx = _normalize_code(idx)
        if norm_idx in alias_map:
            adjusted.loc[idx] = alias_map[norm_idx]
    return adjusted

# =============================================
# 数据处理函数
# =============================================

def generate_gap_factor_views(
        Y_total: pd.DataFrame,
        raw_data: pd.DataFrame,
        current_window_end_idx: int,
    time_offset: int = 1,
    manual_view_overrides: Optional[Dict[str, float]] = None,
    config: Optional[Config] = None,
) -> List[str]:
    """Generate Black-Litterman views from selectable single-factor signals."""
    factor_mode = os.environ.get("BL_FACTOR_MODE", "composite").lower()

    slice_end = current_window_end_idx + 1
    if slice_end <= 0:
        baseline = pd.Series(0.01, index=Y_total.columns)
        return [f"{col} == {baseline[col]:.5f}" for col in baseline.index]

    max_lookback_days = 90
    slice_start = max(0, slice_end - max_lookback_days)

    returns_window = Y_total.iloc[slice_start:slice_end]
    assets = Y_total.columns

    cache_id = id(raw_data)
    cached_id = getattr(generate_gap_factor_views, "_cached_raw_id", None)
    if cached_id != cache_id:
        try:
            close_panel = raw_data['$close'].unstack(level='instrument').sort_index()
        except Exception:
            close_panel = pd.DataFrame()
        try:
            volume_panel = raw_data['$volume'].unstack(level='instrument').sort_index()
        except Exception:
            volume_panel = pd.DataFrame()
        try:
            value_panel = (raw_data['$close'] * raw_data['$volume']).unstack(level='instrument').sort_index()
        except Exception:
            value_panel = pd.DataFrame()
        generate_gap_factor_views._cached_close_panel = close_panel
        generate_gap_factor_views._cached_volume_panel = volume_panel
        generate_gap_factor_views._cached_value_panel = value_panel
        generate_gap_factor_views._cached_raw_id = cache_id
    else:
        close_panel = getattr(generate_gap_factor_views, "_cached_close_panel", pd.DataFrame())
        volume_panel = getattr(generate_gap_factor_views, "_cached_volume_panel", pd.DataFrame())
        value_panel = getattr(generate_gap_factor_views, "_cached_value_panel", pd.DataFrame())

    if isinstance(returns_window.index, pd.Index):
        close_slice = close_panel.reindex(index=returns_window.index, columns=assets)
        volume_slice = volume_panel.reindex(index=returns_window.index, columns=assets)
        value_slice = value_panel.reindex(index=returns_window.index, columns=assets)
    else:
        close_slice = pd.DataFrame(index=returns_window.index, columns=assets, dtype=float)
        volume_slice = pd.DataFrame(index=returns_window.index, columns=assets, dtype=float)
        value_slice = pd.DataFrame(index=returns_window.index, columns=assets, dtype=float)

    close_slice = close_slice.astype(float) if not close_slice.empty else close_slice
    volume_slice = volume_slice.astype(float) if not volume_slice.empty else volume_slice
    value_slice = value_slice.astype(float) if not value_slice.empty else value_slice

    close_ff = close_slice.ffill() if not close_slice.empty else close_slice
    mom_5 = pd.Series(0.0, index=assets, dtype=float)
    mom_21 = pd.Series(0.0, index=assets, dtype=float)
    if not close_ff.empty and close_ff.shape[0] >= 6:
        mom_5 = close_ff.pct_change(5).iloc[-1].reindex(assets)
    if not close_ff.empty and close_ff.shape[0] >= 22:
        mom_21 = close_ff.pct_change(21).iloc[-1].reindex(assets)
    mom_5 = mom_5.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mom_21 = mom_21.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vol_ratio = pd.Series(0.0, index=assets, dtype=float)
    if not volume_slice.empty and volume_slice.shape[0] >= 10:
        vol_ff = volume_slice.replace(0.0, np.nan).ffill()
        vol_ma5 = vol_ff.rolling(5, min_periods=3).mean()
        vol_ma20 = vol_ff.rolling(20, min_periods=10).mean()
        if not vol_ma5.empty and not vol_ma20.empty:
            vol_last5 = vol_ma5.iloc[-1].reindex(assets)
            vol_last20 = vol_ma20.iloc[-1].reindex(assets)
            num = vol_last5.to_numpy(np.float64, copy=False)
            den = vol_last20.to_numpy(np.float64, copy=False)
            vol_div = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=(den != 0))
            vol_ratio = pd.Series(vol_div - 1.0, index=vol_last5.index).reindex(assets)
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    money_flow = pd.Series(0.0, index=assets, dtype=float)
    if not value_slice.empty and value_slice.shape[0] >= 20:
        val_ff = value_slice.replace(0.0, np.nan).ffill()
        val_ma5 = val_ff.rolling(5, min_periods=3).mean()
        val_ma20 = val_ff.rolling(20, min_periods=10).mean()
        if not val_ma5.empty and not val_ma20.empty:
            val_last5 = val_ma5.iloc[-1].reindex(assets)
            val_last20 = val_ma20.iloc[-1].reindex(assets)
            num_v = val_last5.to_numpy(np.float64, copy=False)
            den_v = val_last20.to_numpy(np.float64, copy=False)
            mf_ratio = np.divide(num_v, den_v, out=np.zeros_like(num_v, dtype=float), where=(den_v != 0))
            money_flow = pd.Series(mf_ratio - 1.0, index=val_last5.index).reindex(assets)
    money_flow = money_flow.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    volatility_ratio = pd.Series(0.0, index=assets, dtype=float)
    if returns_window.shape[0] >= 20:
        ret_ff = returns_window.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vol10 = ret_ff.rolling(10, min_periods=5).std(ddof=0)
        vol20 = ret_ff.rolling(20, min_periods=10).std(ddof=0)
        if not vol10.empty and not vol20.empty:
            vol_last10 = vol10.iloc[-1].reindex(assets)
            vol_last20 = vol20.iloc[-1].reindex(assets) + 1e-9
            num_r = vol_last10.to_numpy(np.float64, copy=False)
            den_r = vol_last20.to_numpy(np.float64, copy=False)
            vol_ratio_np = np.divide(num_r, den_r, out=np.zeros_like(num_r, dtype=float), where=(den_r != 0))
            volatility_ratio = pd.Series(vol_ratio_np - 1.0, index=vol_last10.index).reindex(assets)
    volatility_ratio = volatility_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    range_score = pd.Series(0.0, index=assets, dtype=float)
    if not close_ff.empty and close_ff.shape[0] >= 5:
        recent_window = close_ff.iloc[-5:].replace([np.inf, -np.inf], np.nan)
        last_close = close_ff.iloc[-1]
        rolling_min = recent_window.min()
        rolling_max = recent_window.max()
        range_span = (rolling_max - rolling_min).replace(0.0, 1e-9)
        range_raw = (last_close - rolling_min) / range_span
        range_score = range_raw.reindex(assets).fillna(0.0)
    range_score = range_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    base_halflife = 20.0
    min_halflife = 8.0
    max_halflife = 80.0
    eps = 1e-6

    asset_vol = returns_window.std(ddof=0).reindex(assets).fillna(0.0)
    vol_reference = asset_vol.replace(0.0, np.nan).median()
    if not np.isfinite(vol_reference) or vol_reference <= 0:
        vol_reference = float(asset_vol.replace(0.0, np.nan).mean())
    if not np.isfinite(vol_reference) or vol_reference <= 0:
        vol_reference = 1.0

    vol_ratio = (asset_vol / vol_reference).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    ratio_clipped = vol_ratio.clip(lower=0.25, upper=4.0)
    halflife_series = (base_halflife / ratio_clipped).clip(lower=min_halflife, upper=max_halflife)

    vol_boost = (vol_reference / (asset_vol + eps)).clip(lower=0.5, upper=1.5)
    vol_penalty = (asset_vol / (vol_reference + eps)).clip(lower=0.5, upper=1.5)

    halflife_series = halflife_series.reindex(assets).fillna(base_halflife)
    vol_boost = vol_boost.reindex(assets).fillna(1.0)
    vol_penalty = vol_penalty.reindex(assets).fillna(1.0)

    momentum_values = pd.Series(0.0, index=assets, dtype=float)
    reversal_values = pd.Series(0.0, index=assets, dtype=float)
    autocorr_series = pd.Series(0.0, index=assets, dtype=float)

    def _exp_weighted_autocorr(series: pd.Series, halflife: float) -> float:
        if len(series) < 2:
            return 0.0
        decay = np.log(2.0) / max(halflife, eps)
        idx = np.arange(len(series), dtype=float)
        weights = np.exp(-decay * (len(series) - 1 - idx))
        weights = weights / weights.sum()
        values = series.values
        centered = values - np.sum(weights * values)
        pair_weights = weights[1:]
        cov = np.sum(pair_weights * centered[1:] * centered[:-1])
        var_lead = np.sum(pair_weights * centered[:-1] ** 2)
        var_lag = np.sum(pair_weights * centered[1:] ** 2)
        if var_lead <= 0 or var_lag <= 0:
            return 0.0
        return float(np.clip(cov / np.sqrt(var_lead * var_lag), -1.0, 1.0))

    for col in assets:
        series = returns_window[col].dropna()
        if series.empty:
            continue
        hl = float(halflife_series.get(col, base_halflife))
        ema = series.ewm(halflife=hl, adjust=False).mean().iloc[-1]
        last_ret = series.iloc[-1]
        momentum_values.at[col] = ema * vol_boost.get(col, 1.0)
        reversal_values.at[col] = -(last_ret - ema) * vol_penalty.get(col, 1.0)
        autocorr_val = _exp_weighted_autocorr(series, hl)
        autocorr_series.at[col] = autocorr_val * vol_boost.get(col, 1.0)

    features = pd.DataFrame({
        "ema_mom": momentum_values.replace([np.inf, -np.inf], np.nan),
        "reversal": reversal_values.replace([np.inf, -np.inf], np.nan),
        "mom_5": mom_5,
        "mom_21": mom_21,
        "vol_surge": vol_ratio,
        "money_flow": money_flow,
        "volatility_ratio": volatility_ratio,
        "range_score": range_score,
    }).reindex(index=assets)

    def _cs_zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return series.fillna(0.0) * 0.0
        return (series - series.mean()) / std

    zscores = features.apply(_cs_zscore)

    alias_map = {
        "momentum": "ema_mom",
        "momentum_short": "mom_5",
        "momentum_medium": "mom_21",
        "volume": "vol_surge",
        "money_flow": "money_flow",
        "volatility": "volatility_ratio",
        "range": "range_score",
    }
    factor_key = alias_map.get(factor_mode, factor_mode)

    ema_z = zscores.get("ema_mom", pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)
    short_z = zscores.get("mom_5", pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)
    medium_z = zscores.get("mom_21", pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)
    vol_z = zscores.get("vol_surge", pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)
    reversal_z = zscores.get("reversal", pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)
    money_z = zscores.get("money_flow", pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)
    vol_ratio_z = zscores.get("volatility_ratio", pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)
    range_z = zscores.get("range_score", pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)

    autocorr_adj = autocorr_series.reindex(assets).fillna(0.0)
    trend_strength = (
        ema_z.clip(lower=0.0) * 0.45
        + short_z.clip(lower=0.0) * 0.30
        + medium_z.clip(lower=0.0) * 0.15
        + autocorr_adj.clip(lower=0.0) * 0.10
    )
    reversal_strength = (
        (-ema_z).clip(lower=0.0) * 0.30
        + (-short_z).clip(lower=0.0) * 0.30
        + reversal_z.clip(lower=0.0) * 0.30
        + (-autocorr_adj).clip(lower=0.0) * 0.10
    )

    selected = None
    if factor_key in zscores.columns and factor_mode not in {"composite", "adaptive"}:
        selected = zscores[factor_key].fillna(0.0)
        if factor_key == "reversal":
            momentum_weights = pd.Series(0.0, index=assets, dtype=float)
            reversal_weights = pd.Series(1.0, index=assets, dtype=float)
        else:
            momentum_weights = pd.Series(1.0, index=assets, dtype=float)
            reversal_weights = pd.Series(0.0, index=assets, dtype=float)
    else:
        composite = (
            0.55 * ema_z
            + 0.20 * short_z
            + 0.12 * medium_z
            + 0.18 * money_z
            + 0.05 * autocorr_adj
            - 0.08 * vol_z
            - 0.10 * vol_ratio_z
            - 0.12 * range_z
            - 0.15 * reversal_z
        )
        selected = composite.fillna(0.0)
        total_strength = (trend_strength + reversal_strength).replace(0.0, np.nan)
        momentum_weights = trend_strength.divide(total_strength).clip(lower=0.0, upper=1.0).fillna(0.5)
        reversal_weights = reversal_strength.divide(total_strength).clip(lower=0.0, upper=1.0).fillna(0.5)

    selected = selected.fillna(0.0)
    rank_pct = selected.rank(ascending=False, method="average", pct=True)
    rank_pct = rank_pct.fillna(0.5)
    rank_centered = (0.5 - rank_pct) * 2.0
    gap_values = rank_centered.clip(-1.0, 1.0)
    gap_values = _apply_manual_view_overrides(gap_values, manual_view_overrides)

    # 将无量纲的视图映射为日度超额收益预期，提升 BL 辨识度
    default_annual = 0.18
    default_trading_days = 252
    default_cap = np.nan
    if config is not None:
        default_annual = float(getattr(config, "BL_VIEW_ANNUALIZED_RETURN", default_annual))
        default_trading_days = max(1, int(getattr(config, "TRADING_DAYS_PER_YEAR", default_trading_days)))
        default_cap = float(getattr(config, "BL_VIEW_DAILY_CAP", default_cap))
    horizon = 1
    try:
        horizon = max(1, int(abs(time_offset)))
    except Exception:
        horizon = 1
    daily_scale = default_annual / default_trading_days * horizon if default_trading_days > 0 else 0.0
    scaled_views = gap_values * daily_scale
    if np.isfinite(default_cap) and default_cap > 0:
        limit = abs(default_cap)
        scaled_views = scaled_views.clip(lower=-limit, upper=limit)

    regime_score = trend_strength - reversal_strength
    generate_gap_factor_views._last_regime = {
        "momentum_weight": momentum_weights,
        "reversal_weight": reversal_weights,
        "regime_score": regime_score,
        "autocorr": autocorr_series,
        "ema_mom": momentum_values,
        "mom_5": mom_5,
        "mom_21": mom_21,
        "vol_surge": vol_ratio,
        "money_flow": money_flow,
        "volatility_ratio": volatility_ratio,
        "range_score": range_score,
        "trend_strength": trend_strength,
        "reversal_strength": reversal_strength,
        "volatility": asset_vol,
        "halflife": halflife_series,
        "vol_reference": vol_reference,
        "rank_pct": rank_pct,
        "view_values": scaled_views,
        "gap_values": gap_values,
        "scaled_views": scaled_views,
        "daily_scale": daily_scale,
        "view_spread": float(selected.max() - selected.min()) if selected.notna().any() else 0.0,
    }

    gap_views = []
    for column_name, view_value in scaled_views.items():
        if pd.notna(view_value) and np.isfinite(view_value):
            gap_views.append(f"{column_name} == {view_value:.5f}")
        else:
            gap_views.append(f"{column_name} == 0.00000")

    try:
        generate_gap_factor_views._last_factors = {
            "range_score": range_score.copy(),
            "money_flow": money_flow.copy(),
        }
    except Exception:
        generate_gap_factor_views._last_factors = {}

    return gap_views
