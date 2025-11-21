from typing import Set
import pandas as pd
from pathlib import Path
from typing import List, Optional, Iterable, Set, Dict, Union
from collections import defaultdict
import logging
import os
import numpy as np
from config import Config

def _write_csv(path, df, logger: logging.Logger = None, index=True, **to_csv_kwargs):
    """
    Helper: ensure parent dir exists, write dataframe to CSV, log the action.
    path: str or Path
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index, **to_csv_kwargs)
    if logger:
        logger.info("Saved CSV: %s", p)
    else:
        print(f"Saved CSV: {p}")


# 新增：小型 helper，用于把 DataFrame/Series 的索引转为 date（避免重复）
def _index_to_date(obj):
    """
    如果 obj 是 DataFrame 或 Series，返回副本并把索引转为 date（datetime.date）。
    如果为空或其他类型，原样返回。
    """
    if obj is None:
        return obj
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        out = obj.copy()
        out.index = pd.to_datetime(out.index).date
        return out
    return obj

def _normalize_code(code: str) -> str:
    """统一证券代码格式，去除空白并转为大写"""
    return str(code).strip().upper()

def _code_aliases(code: str) -> Set[str]:
    """生成给定证券代码的常见别名集合（含市场前缀与纯数字形式）"""
    norm = _normalize_code(code)
    aliases = {norm}
    if norm.startswith(('SH', 'SZ')) and len(norm) > 2:
        aliases.add(norm[2:])
    elif norm.isdigit() and len(norm) == 6:
        aliases.add(f'SH{norm}')
        aliases.add(f'SZ{norm}')
    return aliases

def _load_cluster_map(path: Union[str, Path, None]) -> Dict[str, str]:
    """读取聚类映射文件，输出代码到风格/行业的映射字典"""
    if not path:
        return {}
    try:
        mapping_path = Path(path)
    except TypeError:
        return {}
    try:
        df = pd.read_csv(mapping_path)
    except FileNotFoundError:
        logging.getLogger(__name__).warning("cluster mapping file not found: %s", mapping_path)
        return {}
    except Exception as exc:
        logging.getLogger(__name__).warning("failed to load cluster mapping file %s: %s", mapping_path, exc)
        return {}
    if df.empty or df.shape[1] < 2:
        return {}
    lower_cols = {col.lower(): col for col in df.columns}
    code_col = next((lower_cols[name] for name in lower_cols if "code" in name or "ticker" in name), df.columns[0])
    cluster_col = next(
        (lower_cols[name] for name in lower_cols if any(keyword in name for keyword in ("cluster", "style", "group", "sector", "category"))),
        None,
    )
    if cluster_col is None:
        candidates = [col for col in df.columns if col != code_col]
        if not candidates:
            return {}
        cluster_col = candidates[0]
    cluster_map: Dict[str, str] = {}
    for code, cluster in df[[code_col, cluster_col]].dropna().itertuples(index=False):
        cluster_id = str(cluster).strip()
        if not cluster_id:
            continue
        for alias in _code_aliases(code):
            cluster_map[alias] = cluster_id
    return cluster_map

def _prune_small_weights(
    weights: pd.DataFrame,
    threshold: float,
    cluster_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """剔除低于阈值的权重，并按聚类或全局规则回收并归一"""
    threshold = float(threshold)
    if weights.empty or not np.isfinite(threshold) or threshold <= 0.0:
        return weights
    pruned = weights.copy()
    if cluster_map:
        cluster_series = pd.Series(
            {col: cluster_map.get(_normalize_code(col), cluster_map.get(col)) for col in weights.columns},
            index=weights.columns,
            dtype=object,
        )
    else:
        cluster_series = pd.Series({col: None for col in weights.columns}, index=weights.columns, dtype=object)
    for dt in pruned.index:
        row = pruned.loc[dt].astype(float).clip(lower=0.0)
        total = row.sum()
        if total <= 0:
            continue
        row = row / total
        keep_mask = row >= threshold
        drop_mask = (row > 0) & (~keep_mask)
        keep_sum = row[keep_mask].sum()
        new_row = pd.Series(0.0, index=row.index)
        if keep_sum <= 0:
            idxmax = row.idxmax()
            if pd.notna(idxmax):
                new_row[idxmax] = 1.0
            pruned.loc[dt] = new_row
            continue

        new_row.loc[keep_mask] = row.loc[keep_mask]
        drop_totals = defaultdict(float)
        for code in row.index[drop_mask]:
            cluster_id = cluster_series.get(code, None)
            drop_totals[cluster_id] += row[code]

        global_remainder = 0.0
        for cluster_id, drop_weight in drop_totals.items():
            eligible_mask = keep_mask & (cluster_series == cluster_id)
            if eligible_mask.any():
                eligible_weights = row.loc[eligible_mask]
                weight_sum = eligible_weights.sum()
                if weight_sum > 0:
                    share = eligible_weights / weight_sum
                    new_row.loc[eligible_mask] += drop_weight * share
                else:
                    global_remainder += drop_weight
            else:
                global_remainder += drop_weight

        if global_remainder > 0:
            eligible_mask = keep_mask
            eligible_weights = row.loc[eligible_mask]
            weight_sum = eligible_weights.sum()
            if weight_sum > 0:
                share = eligible_weights / weight_sum
                new_row.loc[eligible_mask] += global_remainder * share
            else:
                idxmax = row.idxmax()
                new_row[:] = 0.0
                if pd.notna(idxmax):
                    new_row[idxmax] = 1.0
                pruned.loc[dt] = new_row
                continue

        new_sum = new_row.sum()
        if new_sum > 0:
            new_row = new_row / new_sum
        else:
            idxmax = row.idxmax()
            new_row[:] = 0.0
            if pd.notna(idxmax):
                new_row[idxmax] = 1.0
        pruned.loc[dt] = new_row
    return pruned

def _compute_blend_series(series: pd.Series, threshold: float, floor: float, blend_min: float, blend_max: float) -> pd.Series:
    """对序列逐元素应用混合比例计算，生成防守触发强度序列"""
    if series.empty:
        return pd.Series(dtype=float)
    return series.apply(lambda val: _compute_blend_value(val, threshold, floor, blend_min, blend_max))

def _compute_blend_value(value: float, threshold: float, floor: float, blend_min: float, blend_max: float) -> float:
    """根据严重度在给定范围内插值出混合比例"""
    severity = _compute_regime_severity(value, threshold, floor)
    if severity <= 0.0:
        return 0.0
    blend_min = float(np.clip(blend_min, 0.0, 1.0))
    blend_max = float(np.clip(blend_max, 0.0, 1.0))
    if blend_max < blend_min:
        blend_min, blend_max = blend_max, blend_min
    return float(blend_min + severity * (blend_max - blend_min))

def _compute_regime_severity(value: float, threshold: float, floor: float) -> float:
    """计算指标越过阈值后的严重度（0-1），用于风险防守调节"""
    if not np.isfinite(value) or not np.isfinite(threshold):
        return 0.0
    if not np.isfinite(floor):
        floor = threshold - 0.25
    if floor >= threshold:
        floor = threshold - 1e-6
    denom = threshold - floor
    if denom <= 0:
        denom = 1.0
    severity = (threshold - value) / denom
    return float(np.clip(severity, 0.0, 1.0))

def _cap_weights(weights: pd.Series, cap: float) -> pd.Series:
    """强制约束单一权重上限并重新归一化"""
    cap = float(cap)
    if weights.empty:
        return weights
    clipped = weights.clip(lower=0.0).astype(float)
    total = clipped.sum()
    if total <= 0:
        return weights
    clipped /= total
    if cap <= 0.0 or cap >= 1.0:
        return clipped
    epsilon = 1e-12
    while True:
        over_mask = clipped > cap + epsilon
        if not over_mask.any():
            break
        deficit = (clipped[over_mask] - cap).sum()
        clipped[over_mask] = cap
        remain_mask = ~over_mask
        remain_sum = clipped[remain_mask].sum()
        if remain_sum <= epsilon:
            remain_indices = clipped.index[remain_mask]
            if len(remain_indices) == 0:
                break
            redistributed = deficit / len(remain_indices)
            clipped.loc[remain_indices] = redistributed
        else:
            clipped.loc[remain_mask] += clipped[remain_mask] / remain_sum * deficit
    clipped = clipped.clip(lower=0.0)
    final_sum = clipped.sum()
    if final_sum > 0:
        clipped /= final_sum
    return clipped

def parse_manual_view_override_string(text: str) -> Dict[str, float]:
    """解析命令行传入的 'CODE=value' 字符串为观点覆盖字典"""
    overrides: Dict[str, float] = {}
    if not text:
        return overrides
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk or '=' not in chunk:
            continue
        code, value = chunk.split('=', 1)
        code = code.strip()
        value = value.strip()
        if not code or not value:
            continue
        try:
            overrides[_normalize_code(code)] = float(value)
        except ValueError:
            continue
    return overrides

def _parse_view_strings(view_strings: Iterable[str]) -> Dict[str, float]:
    """将形如 'CODE == value' 的字符串解析为观点字典"""
    views: Dict[str, float] = {}
    for item in view_strings or []:
        if '==' not in item:
            continue
        left, right = item.split('==', 1)
        code = left.strip()
        try:
            value = float(right.strip())
        except ValueError:
            try:
                value = float(right.strip().split()[0])
            except Exception:
                continue
        views[code] = value
    return views

def _build_baseline_series(assets: Iterable[str], config: Config) -> pd.Series:
    """创建基准权重Series，若无配置则回退至等权"""
    return _build_weight_series(
        assets,
        getattr(config, "BENCHMARK_WEIGHTS", {}),
        fallback_equal=True,
    )

def _build_weight_series(
    assets: Iterable[str],
    weight_map: Optional[Dict[str, float]],
    *,
    fallback_equal: bool = False,
) -> pd.Series:
    """根据权重字典生成与资产序列对齐的权重Series，必要时回退等权"""
    assets_index = pd.Index(list(assets))
    series = pd.Series(0.0, index=assets_index, dtype=float)
    if not weight_map:
        if fallback_equal and len(series) > 0:
            series[:] = 1.0 / len(series)
        return series
    alias_map: Dict[str, float] = {}
    for code, weight in weight_map.items():
        try:
            w = float(weight)
        except Exception:
            continue
        for alias in _code_aliases(code):
            alias_map[alias] = alias_map.get(alias, 0.0) + w
    total = 0.0
    for asset in assets_index:
        norm_asset = _normalize_code(asset)
        if norm_asset in alias_map:
            series.loc[asset] = alias_map[norm_asset]
            total += alias_map[norm_asset]
    if total > 0:
        series /= total
    elif fallback_equal and len(series) > 0:
        series[:] = 1.0 / len(series)
    return series

def _apply_sticky_weights(weights: pd.DataFrame, config: Config) -> pd.DataFrame:
    """对权重时间序列应用粘性系数，平滑持仓变动"""
    stick_alpha = float(getattr(config, "STICKY_WEIGHT_ALPHA", 0.0))
    if weights.empty or not np.isfinite(stick_alpha) or stick_alpha <= 0.0:
        return weights
    stick_alpha = float(np.clip(stick_alpha, 0.0, 1.0))
    smoothed = weights.sort_index().copy()
    prev = None
    for dt in smoothed.index:
        row = smoothed.loc[dt].astype(float).clip(lower=0.0)
        total = row.sum()
        if total > 0:
            row = row / total
        if prev is not None and stick_alpha > 0.0:
            mixed = row * (1.0 - stick_alpha) + prev * stick_alpha
            mixed = mixed.clip(lower=0.0)
            mixed_sum = mixed.sum()
            row = mixed / mixed_sum if mixed_sum > 0 else prev.copy()
        smoothed.loc[dt] = row
        prev = row
    return smoothed