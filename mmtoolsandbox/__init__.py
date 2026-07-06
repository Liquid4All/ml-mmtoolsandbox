# Copyright © 2026 Apple Inc.

import os
import warnings

# We need to hard-code the version number here instead of e.g. reading it from the
# `pyproject.toml` file because when this package gets installed somewhere else the
# `pyproject.toml` file is not included.
__version__ = "1.0.0"
# Hack for setuptools and pip import issue. Kept here to ensure this happens before other imports.
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
warnings.filterwarnings(
    "ignore", message="Mixing V1 models and V2 models*", category=UserWarning
)
