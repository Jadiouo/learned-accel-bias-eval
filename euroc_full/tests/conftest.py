# -*- coding: utf-8 -*-
"""讓 `pytest` 在未 `pip install -e .` 時也能 import euroc_full（src layout）。"""
import os
import sys

SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
