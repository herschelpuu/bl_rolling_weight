
from utils import *
from utils import _code_aliases
from config import Config
from qlib.data import D

def load_instrument_codes(path: str) -> List[str]:
    """读取标的文件，忽略注释与空行，返回代码列表"""
    codes: List[str] = []
    if not path or not os.path.exists(path):
        return codes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if parts:
                codes.append(parts[0])
    return codes

def _build_drop_set(codes: Optional[Iterable[str]]) -> Set[str]:
    """将待剔除代码列表扩展为包含所有别名的集合"""
    drop_set: Set[str] = set()
    for code in codes or []:
        drop_set.update(_code_aliases(code))
    return drop_set

class UniverseBuilder:
    """统一管理标的集合的加载、过滤与兜底逻辑"""

    def __init__(
        self,
        config: Config,
        drop_codes: Optional[Iterable[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Store configuration, drop list, and logger for later reuse."""
        self.config = config
        self.drop_codes = list(drop_codes or [])
        self.logger = logger or logging.getLogger(__name__)

    def build(self) -> List[str]:
        """Load instruments from file with fallbacks and apply manual drop rules."""
        print(f"加载标的列表文件: {self.config.file_path}")
        stockpool = load_instrument_codes(self.config.file_path)
        if not stockpool:
            self._log("instrument file %s 为空，尝试从 qlib 直接加载", self.config.file_path, level="warning")
            stockpool = self._load_from_qlib()
        self._log("原始标的数量: %d", len(stockpool))
        effective_drops = _build_drop_set(self.drop_codes) | _build_drop_set(self.config.MANUAL_DROP_CODES)
        if effective_drops:
            stockpool = self._apply_drop_list(stockpool, effective_drops)
        if not stockpool:
            raise RuntimeError("标的列表为空，无法继续生成权重")
        return stockpool

    def _load_from_qlib(self) -> List[str]:
        """Fetch the instrument universe directly from qlib when file source fails."""
        try:
            instruments = D.instruments(market=self.config.MARKET)
        except Exception as exc:  # pragma: no cover
            self._log("无法加载市场 %s: %s", self.config.MARKET, exc, level="error")
            return []
        if isinstance(instruments, (list, tuple, set)):
            return list(instruments)
        if isinstance(instruments, dict):
            pipe = instruments.get('filter_pipe') or []
            if pipe and isinstance(pipe[0], list):
                return pipe[0]
        return list(instruments)

    def _apply_drop_list(self, stockpool: List[str], drop_aliases: Set[str]) -> List[str]:
        """Filter out instruments whose aliases intersect with the configured drop set."""
        kept, dropped = [], []
        for code in stockpool:
            aliases = _code_aliases(code)
            if aliases & drop_aliases:
                dropped.append(code)
            else:
                kept.append(code)
        if dropped:
            self._log("排除指定代码: %s", dropped)
        return kept

    def _log(self, msg: str, *args, level: str = "info") -> None:
        """Proxy logging helper that respects severity levels."""
        if level == "warning":
            self.logger.warning(msg, *args)
        elif level == "error":
            self.logger.error(msg, *args)
        else:
            self.logger.info(msg, *args)
