import pandas as pd
from typing import List, Optional, Dict, Tuple, Iterable
import numpy as np
from config import Config
from utils import _normalize_code, _code_aliases, _cap_weights, \
    _compute_blend_series, _compute_regime_severity, _compute_blend_value
from utils import _build_baseline_series, _build_weight_series

def _apply_spearman_guard(
    weights: pd.DataFrame,
    views: pd.DataFrame,
    raw_data: pd.DataFrame,
    config: Config,
    factor_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Spearman-based safeguards and defensive overlays before publishing weights."""
    if weights.empty or views.empty:
        return weights, pd.DataFrame()

    adjusted_weights = weights.copy()
    weight_idx = pd.to_datetime(adjusted_weights.index)
    if isinstance(weight_idx, pd.DatetimeIndex):
        if weight_idx.tz is not None:
            weight_idx = weight_idx.tz_convert(None)
        weight_idx = weight_idx.normalize()
        adjusted_weights.index = weight_idx

    views = views.copy()
    view_idx = pd.to_datetime(views.index)
    if isinstance(view_idx, pd.DatetimeIndex):
        if view_idx.tz is not None:
            view_idx = view_idx.tz_convert(None)
        view_idx = view_idx.normalize()
        views.index = view_idx
    use_baseline = bool(getattr(config, "USE_GUARD_BASELINE", True))
    if use_baseline:
        baseline = _build_baseline_series(adjusted_weights.columns, config)
        if baseline.sum() == 0 and len(baseline) > 0:
            baseline[:] = 1.0 / len(baseline)
        baseline = baseline / baseline.sum()
    else:
        baseline = pd.Series(0.0, index=adjusted_weights.columns, dtype=float)
    defensive_series = _build_weight_series(adjusted_weights.columns, getattr(config, "DEFENSIVE_WEIGHTS", {}))
    guard_def_threshold = float(getattr(config, "GUARD_DEFENSIVE_VIEW_THRESHOLD", np.nan))
    guard_def_floor = float(getattr(config, "GUARD_DEFENSIVE_VIEW_FLOOR", guard_def_threshold - 0.1))
    guard_def_blend_min = float(np.clip(getattr(config, "GUARD_DEFENSIVE_BLEND_MIN", getattr(config, "GUARD_DEFENSIVE_BLEND", 0.35)), 0.0, 1.0))
    guard_def_blend_max = float(np.clip(getattr(config, "GUARD_DEFENSIVE_BLEND_MAX", getattr(config, "GUARD_DEFENSIVE_BLEND", 0.6)), 0.0, 1.0))
    if guard_def_blend_max < guard_def_blend_min:
        guard_def_blend_min, guard_def_blend_max = guard_def_blend_max, guard_def_blend_min
    guard_def_blend_default = float(np.clip(getattr(config, "GUARD_DEFENSIVE_BLEND", 0.0), 0.0, 1.0))
    has_defensive = defensive_series.sum() > 0 and np.isfinite(guard_def_threshold)

    try:
        close_panel = raw_data['$close'].unstack(level='instrument')
        close_panel.index = pd.to_datetime(close_panel.index)
        if isinstance(close_panel.index, pd.DatetimeIndex) and close_panel.index.tz is not None:
            close_panel.index = close_panel.index.tz_convert(None)
        close_panel.index = close_panel.index.normalize()
    except Exception:
        close_panel = pd.DataFrame()
    forward_returns = pd.DataFrame()
    if not close_panel.empty:
        forward_returns = close_panel.pct_change().shift(-1)
        forward_returns.index = forward_returns.index.normalize()

    factor_frames = factor_frames or {}

    def _prepare_factor_frame(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if frame is None or frame.empty:
            return None
        prepared = frame.copy()
        idx = pd.to_datetime(prepared.index)
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            idx = idx.tz_convert(None)
        prepared.index = idx.normalize()
        return prepared

    range_frame = _prepare_factor_frame(factor_frames.get("range_score"))
    money_frame = _prepare_factor_frame(factor_frames.get("money_flow"))

    guard_window = max(1, int(getattr(config, "SPEARMAN_GUARD_WINDOW", 5)))
    guard_min_obs = max(1, int(getattr(config, "SPEARMAN_GUARD_MIN_OBS", 3)))
    guard_threshold = float(getattr(config, "SPEARMAN_GUARD_THRESHOLD", 0.0))
    guard_active_scale = float(getattr(config, "SPEARMAN_GUARD_ACTIVE_SCALE", 0.5))
    guard_active_scale = float(np.clip(guard_active_scale, 0.0, 1.0))
    pos_threshold = float(getattr(config, "SPEARMAN_POS_THRESHOLD", 0.0))
    pos_scale = float(getattr(config, "SPEARMAN_POS_SCALE", 0.3))
    pos_scale = float(np.clip(pos_scale, 0.0, 1.0))
    pos_power = float(getattr(config, "SPEARMAN_POS_POWER", 1.5))
    if pos_power < 1.0:
        pos_power = 1.0

    guard_top_n = max(1, int(getattr(config, "GUARD_RANGE_TOP_N", 10)))
    range_threshold = float(getattr(config, "GUARD_RANGE_THRESHOLD", 0.9))
    range_ratio_req = float(np.clip(getattr(config, "GUARD_RANGE_RATIO", 0.5), 0.0, 1.0))
    range_blend_target_cfg = float(np.clip(getattr(config, "GUARD_RANGE_BLEND", 0.0), 0.0, 1.0))
    range_blend_cap_cfg = float(np.clip(getattr(config, "GUARD_RANGE_BLEND_MAX", 1.0), 0.0, 1.0))
    range_scale_cfg = float(np.clip(getattr(config, "GUARD_RANGE_SCALE", guard_active_scale), 0.0, 1.0))
    range_cap_cfg = float(getattr(config, "GUARD_RANGE_CAP", getattr(config, "MAX_WEIGHT_CAP", 1.0)))
    money_threshold = float(getattr(config, "GUARD_MONEYFLOW_THRESHOLD", -0.15))
    money_neg_ratio_req = float(np.clip(getattr(config, "GUARD_MONEYFLOW_NEG_RATIO", 0.6), 0.0, 1.0))
    money_scale_cfg = float(np.clip(getattr(config, "GUARD_MONEYFLOW_SCALE", guard_active_scale), 0.0, 1.0))

    history: List[float] = []
    guard_records: List[Dict[str, object]] = []

    for dt in adjusted_weights.index.sort_values():
        dt_key = pd.Timestamp(dt).normalize()
        weight_row = adjusted_weights.loc[dt_key].astype(float).copy()
        if use_baseline:
            baseline_row = baseline.reindex(weight_row.index).fillna(0.0)
        else:
            baseline_row = weight_row.copy()
        defensive_row = defensive_series.reindex(weight_row.index).fillna(0.0) if has_defensive else pd.Series(0.0, index=weight_row.index)
        view_row = pd.Series(dtype=float)
        view_mean = np.nan
        if dt_key in views.index:
            view_row = views.loc[dt_key].astype(float)
            view_mean = view_row.replace([np.inf, -np.inf], np.nan).dropna().mean()
        guard_def_blend_value = 0.0
        guard_def_severity = 0.0
        down_signal = False
        if has_defensive and np.isfinite(view_mean):
            guard_def_severity = _compute_regime_severity(view_mean, guard_def_threshold, guard_def_floor)
            if guard_def_severity > 0.0:
                guard_def_blend_value = _compute_blend_value(
                    view_mean,
                    guard_def_threshold,
                    guard_def_floor,
                    guard_def_blend_min,
                    guard_def_blend_max,
                )
                if guard_def_blend_value <= 0.0 and guard_def_blend_default > 0.0:
                    guard_def_blend_value = guard_def_blend_default * guard_def_severity
                down_signal = guard_def_blend_value > 0.0
        if down_signal:
            base_anchor = baseline_row if use_baseline else weight_row
            mixed_base = base_anchor * (1.0 - guard_def_blend_value) + defensive_row * guard_def_blend_value
            mixed_base = mixed_base.clip(lower=0.0)
            base_sum = mixed_base.sum()
            if base_sum > 0:
                baseline_row = mixed_base / base_sum
            else:
                down_signal = False
        valid_history = [h for h in history if h is not None and np.isfinite(h)]
        guard_avg = float(np.mean(valid_history[-guard_window:])) if len(valid_history) >= guard_min_obs else np.nan
        guard_trigger = False
        scale_used = 1.0
        pos_boost = False
        pos_scale_used = 0.0

        range_series = range_frame.loc[dt_key].astype(float) if range_frame is not None and dt_key in range_frame.index else None
        money_series = money_frame.loc[dt_key].astype(float) if money_frame is not None and dt_key in money_frame.index else None

        top_n = min(len(weight_row), guard_top_n)
        top_assets = weight_row.nlargest(top_n).index if top_n > 0 else weight_row.index

        range_ratio_val = np.nan
        range_trigger = False
        if range_series is not None and len(top_assets) > 0:
            top_range = range_series.reindex(top_assets)
            high_mask = top_range >= range_threshold
            high_count = int(high_mask.sum())
            range_ratio_val = high_count / len(top_assets) if len(top_assets) > 0 else np.nan
            if np.isfinite(range_ratio_val) and range_ratio_val >= range_ratio_req:
                range_trigger = True

        money_avg = np.nan
        money_neg_ratio_val = np.nan
        money_trigger = False
        if money_series is not None and len(top_assets) > 0:
            top_money = money_series.reindex(top_assets)
            money_avg = top_money.mean()
            neg_mask = top_money <= money_threshold
            money_neg_ratio_val = neg_mask.sum() / len(top_assets) if len(top_assets) > 0 else np.nan
            if (np.isfinite(money_avg) and money_avg <= money_threshold) or (np.isfinite(money_neg_ratio_val) and money_neg_ratio_val >= money_neg_ratio_req):
                money_trigger = True

        if np.isfinite(guard_avg) and guard_avg < guard_threshold:
            guard_trigger = True
            scale_used = min(scale_used, guard_active_scale)

        if not guard_trigger and np.isfinite(guard_avg) and guard_avg >= pos_threshold and pos_scale > 0.0:
            active = weight_row.clip(lower=0.0)
            total_active = active.sum()
            if total_active > 0:
                active = active / total_active
                enhanced = np.power(active, pos_power)
                enhanced_sum = enhanced.sum()
                if enhanced_sum > 0:
                    enhanced = enhanced / enhanced_sum
                    pos_boost = True
                    pos_scale_used = pos_scale
                    weight_row = weight_row * (1.0 - pos_scale_used) + enhanced * pos_scale_used
                    weight_row = weight_row.clip(lower=0.0)
                    total = weight_row.sum()
                    if total > 0:
                        weight_row = weight_row / total

        dynamic_max_cap = float(getattr(config, "MAX_WEIGHT_CAP", 1.0))

        if range_trigger:
            guard_trigger = True
            scale_used = min(scale_used, range_scale_cfg)
            guard_def_blend_value = min(max(guard_def_blend_value, range_blend_target_cfg), range_blend_cap_cfg)
            if guard_def_blend_value > 0:
                down_signal = True
            guard_def_severity = max(guard_def_severity, range_ratio_val if np.isfinite(range_ratio_val) else guard_def_severity)
            if 0.0 < range_cap_cfg < dynamic_max_cap:
                dynamic_max_cap = range_cap_cfg

        if money_trigger:
            guard_trigger = True
            scale_used = min(scale_used, money_scale_cfg)
            if guard_def_blend_value <= 0.0 and guard_def_blend_min > 0.0:
                guard_def_blend_value = guard_def_blend_min
            if guard_def_blend_value > 0.0 and has_defensive:
                down_signal = True

        if guard_trigger and scale_used < 1.0:
            if use_baseline:
                mixed = baseline_row * (1.0 - scale_used) + weight_row * scale_used
                mixed = mixed.clip(lower=0.0)
                total = mixed.sum()
                weight_row = mixed / total if total > 0 else baseline_row.copy()
            elif has_defensive and defensive_row.sum() > 0:
                fallback = defensive_row.clip(lower=0.0)
                fallback_sum = fallback.sum()
                if fallback_sum > 0:
                    fallback = fallback / fallback_sum
                    mixed = fallback * (1.0 - scale_used) + weight_row * scale_used
                    mixed = mixed.clip(lower=0.0)
                    total = mixed.sum()
                    weight_row = mixed / total if total > 0 else fallback.copy()

        if down_signal:
            mixed_def = weight_row * (1.0 - guard_def_blend_value) + defensive_row * guard_def_blend_value
            mixed_def = mixed_def.clip(lower=0.0)
            mixed_sum = mixed_def.sum()
            if mixed_sum > 0:
                weight_row = mixed_def / mixed_sum
            else:
                down_signal = False

        if 0.0 < dynamic_max_cap < 1.0:
            weight_row = _cap_weights(weight_row, dynamic_max_cap)

        adjusted_weights.loc[dt_key] = weight_row

        spearman_val = np.nan
        if not view_row.empty and not forward_returns.empty and dt_key in forward_returns.index:
            return_row = forward_returns.loc[dt_key].astype(float)
            aligned = pd.concat([
                view_row.rename('view'),
                return_row.rename('fwd_ret')
            ], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if len(aligned) >= 3:
                view_rank = aligned['view'].rank(pct=True, method='average')
                ret_rank = aligned['fwd_ret'].rank(pct=True, method='average')
                spearman_val = view_rank.corr(ret_rank, method='pearson')

        history.append(spearman_val)
        guard_records.append({
            'datetime': dt_key,
            'guard_avg': guard_avg,
            'guard_trigger': guard_trigger,
            'guard_scale': scale_used,
            'spearman_realized': spearman_val,
            'pos_boost': pos_boost,
            'pos_scale': pos_scale_used,
            'defensive_overlay': down_signal,
            'defensive_blend': guard_def_blend_value if down_signal else 0.0,
            'defensive_severity': guard_def_severity if down_signal else 0.0,
            'range_overheat': range_trigger,
            'range_ratio_top': range_ratio_val,
            'money_outflow': money_trigger,
            'money_avg_top': money_avg,
            'money_neg_ratio_top': money_neg_ratio_val,
            'range_cap_applied': dynamic_max_cap,
        })

    guard_df = pd.DataFrame.from_records(guard_records).set_index('datetime') if guard_records else pd.DataFrame()
    return adjusted_weights, guard_df




