"""Top-level package for CvUtils."""

from . import core, highgui, img_processing, visualization
from ._version import get_versions
from .core import *  # noqa: F401, F403
from .highgui import *  # noqa: F401, F403
from .img_processing import *  # noqa: F401, F403
from .visualization import *  # noqa: F401, F403

__author__ = """Deyu He"""
__email__ = "18565286660@163.com"
__version__ = get_versions()["version"]
del get_versions

__all__ = []
__all__ += img_processing.__all__
__all__ += highgui.__all__
__all__ += visualization.__all__
__all__ += core.__all__
