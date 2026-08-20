#!/usr/bin/env python3
"""Notebook display helpers: render the plots and tables the pipeline produced.

Missing files are skipped silently, so a partial run renders the plots it has.

Import the functions, not the module: `display` is also the name of IPython's
own display() function that the notebooks call directly.
"""

import os

import pandas as pd
from IPython.display import HTML, Image
from IPython.display import display as _display

DEFAULT_WIDTH = 820


def header(text, level=3):
    _display(HTML(f"<h{level} style='border-bottom:1px solid #999;"
                  f"margin-top:1em'>{text}</h{level}>"))


def _caption(text):
    _display(HTML(f"<div style='color:#555;font-size:0.9em'>{text}</div>"))


def show(path, width=DEFAULT_WIDTH, caption=None):
    """Display one image if it exists; return True when shown."""
    if not os.path.exists(path):
        return False
    if caption:
        _caption(caption)
    _display(Image(filename=path, width=width))
    return True


def _normalize(items, base=None):
    """(path, caption) pairs for *items*, resolved against *base*."""
    pairs = [(p, None) if isinstance(p, str) else tuple(p) for p in items]
    if base:
        pairs = [(os.path.join(base, p), c) for p, c in pairs]
    return pairs


def show_all(items, base=None, width=DEFAULT_WIDTH, titles=False, level=3):
    """Display every image in *items* that exists; return how many were shown.

    titles: also emit a header per image, taken from its file name.
    """
    shown = 0
    for path, caption in _normalize(items, base):
        if not os.path.exists(path):
            continue
        if titles:
            header(os.path.basename(path), level)
        shown += show(path, width=width, caption=caption)
    return shown


def show_group(title, items, base=None, width=DEFAULT_WIDTH, level=3):
    """Show a titled group of plots; the title is skipped when nothing exists."""
    pairs = _normalize(items, base)
    if not any(os.path.exists(p) for p, _ in pairs):
        return False
    header(title, level)
    show_all(pairs, width=width)
    return True


def show_table(path, caption=None, sep="\t"):
    """Display a delimited table as a DataFrame if it exists; True when shown."""
    if not os.path.exists(path):
        return False
    if caption:
        _caption(caption)
    _display(pd.read_csv(path, sep=sep))
    return True


def method_plot(ds, method_key, rel, variant, root="out"):
    """Path to a per-dataset, per-method analysis plot.

    The reference segmentation is not matched into a variant, so it lives
    outside the variant dir.
    """
    if method_key == "ref":
        return f"{root}/{ds}/ref/{rel}"
    return f"{root}/{ds}/{variant}/{method_key}/{rel}"
