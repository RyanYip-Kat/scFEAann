"""Programmatic matplotlib layout checks used before figure export."""

from __future__ import annotations

import logging
import warnings

import matplotlib.text as mtext


_GLYPH_MARKERS = ("missing from", "Glyph", "findfont")


class _GlyphHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if any(marker in message for marker in _GLYPH_MARKERS):
            self.messages.append(message)


def _visible_texts(fig) -> list:
    return [
        text
        for text in fig.findobj(mtext.Text)
        if text.get_visible() and text.get_text().strip()
    ]


def _tick_overlap(axis, renderer) -> bool:
    for labels in (axis.get_xticklabels(), axis.get_yticklabels()):
        visible = [
            label for label in labels
            if label.get_visible() and label.get_text().strip()
        ]
        boxes = [label.get_window_extent(renderer) for label in visible]
        for first, second in zip(boxes, boxes[1:], strict=False):
            if first.overlaps(second):
                return True
    return False


def audit_layout(fig) -> list[tuple[str, str]]:
    """Check missing glyphs, clipped text, and overlapping tick labels."""
    issues: list[tuple[str, str]] = []
    handler = _GlyphHandler()
    logger = logging.getLogger("matplotlib")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    warning_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        warning_messages = [
            str(item.message)
            for item in caught
            if any(marker in str(item.message) for marker in _GLYPH_MARKERS)
        ]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    glyph_messages = warning_messages + handler.messages
    if glyph_messages:
        issues.append(
            (
                "FAIL",
                f"Missing glyphs detected: {glyph_messages[0][:240]}",
            )
        )

    figure_width = float(fig.bbox.width)
    figure_height = float(fig.bbox.height)
    clipped: list[str] = []
    tick_ids = {
        id(label)
        for axis in fig.axes
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels())
    }
    for text in _visible_texts(fig):
        if id(text) in tick_ids:
            continue
        box = text.get_window_extent(renderer)
        if (
            box.x0 < -2
            or box.y0 < -2
            or box.x1 > figure_width + 2
            or box.y1 > figure_height + 2
        ):
            clipped.append(text.get_text().replace("\n", " ")[:30])
    if clipped:
        issues.append(
            (
                "WARN",
                f"Text may be clipped outside the canvas: {list(dict.fromkeys(clipped))[:6]}",
            )
        )

    overlap_count = sum(
        1 for axis in fig.axes if _tick_overlap(axis, renderer)
    )
    if overlap_count:
        issues.append(
            (
                "WARN",
                f"{overlap_count} axes contain overlapping tick labels.",
            )
        )
    return issues
