from logging import Formatter
from logging import StreamHandler
from logging import getLogger


__version__ = "0.1.0"

logger = getLogger(__name__)
fmt = Formatter(
    "[%(levelname)s] %(name)s %(asctime)s - %(filename)s: %(lineno)d: %(message)s",
)
sh = StreamHandler()
sh.setLevel("DEBUG")
sh.setFormatter(fmt)
logger.addHandler(sh)
logger.setLevel("WARN")
