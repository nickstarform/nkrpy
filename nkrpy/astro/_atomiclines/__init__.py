"""Atomic Lines Manipulator.

This module holds the generic linelist and the class
    to handle manipulation of atomic lines.

The file linelist holds the lines themselves and can be
accessed from `atomiclines` while the converter is `Lines`
"""
from ._atomiclines.__linelist import atomiclines as linelist
from ._atomiclines.lines import Lines

__all__ = ['linelist', 'Lines']
