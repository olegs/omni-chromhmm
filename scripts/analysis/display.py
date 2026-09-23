#!/usr/bin/env python3
"""Notebook display helpers: render the plots and tables the pipeline produced.

Every plot an analysis notebook shows was written to disk by a computation cell
and is read straight back off disk here. Missing files are skipped silently, so
a notebook renders cleanly regardless of which segmentations were actually
produced — a partial run shows the plots it has instead of raising.

Public API
----------
header(text, level=3)              → titled section separator
show(path, ...)                    → one image, True when it existed
show_all(items, base=None, ...)    → every image that exists, count shown
show_group(title, items, ...)      → titled group, title skipped when empty
show_table(path, ...)              → a TSV as a DataFrame, True when it existed
method_plot(ds, method, rel, variant) → path to a per-method analysis plot

Usage (from a notebook, with scripts/analysis on sys.path):
    from display import header, show, show_all, show_group, show_table

Import the functions, not the module: `display` is also the name of IPython's
own display() function that the notebooks call directly, and `import display`
would shadow it.
"""

import os

import pandas as pd
from IPython.display import HTML, Image
from IPython.display import display as _display

# Wide enough to read axis labels on a typical screen without scrolling.
DEFAULT_WIDTH = 820


def header(text, level=3):
    """Render *text* as a section separator above the plots that follow."""
    _display(HTML(f"<h{level} style='border-bottom:1px solid #999;"
                  f"margin-top:1em'>{text}</h{level}>"))


def _caption(text):
    """Render small grey explanatory text above a plot or table."""
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
    """(path, caption) pairs for *items*, resolved against *base*.

    Each item is either a path or a (path, caption) tuple; *base* lets a caller
    list bare file names for one output directory.
    """
    pairs = [(p, None) if isinstance(p, str) else tuple(p) for p in items]
    if base:
        pairs = [(os.path.join(base, p), c) for p, c in pairs]
    return pairs


def show_all(items, base=None, width=DEFAULT_WIDTH, titles=False, level=3):
    """Display every image in *items* that exists; return how many were shown.

    titles: also emit a header per image, taken from its file name — for
    listings where each plot needs its own label rather than one group title.
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
    """Show a titled group of plots; the title is skipped when nothing exists.

    Returns True when at least one plot was shown, so a caller can fall back to
    a note about missing inputs.
    """
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
    outside the variant dir (<root>/<ds>/ref/... rather than
    <root>/<ds>/<variant>/ref/...).
    """
    if method_key == "ref":
        return f"{root}/{ds}/ref/{rel}"
    return f"{root}/{ds}/{variant}/{method_key}/{rel}"
