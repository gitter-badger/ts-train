__doc__ = """
ts-train - Time-Series Data Analysis, Manipulation and Machine Learning Modeling Library for Python
===================================================================================================

Documentation is available in the docstrings and online at https://minesh1291.github.io/ts-train

Contents
--------
::

 - Data Preparation
    - Validation
    - Cleaning
        - Missing value imputation
        - Reindex
        - Resampling
    - Normalization
    - Scaling
    - Transform
    - Encoding
    - Embedding
 - Feature Extraction

Subpackages
-----------

data --- Data Manipulation


"""

__author__ = "Minesh A. Jethva"
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "0.2"
#  __date__ = "30 May, 2020"
__maintainer__ = "Minesh A. Jethva"
__email__ = "minesh.1291@gmail.com"
__status__ = "Development Status :: 2 - Pre-Alpha"


__all__ = ["data"]

from . import data
