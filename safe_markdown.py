"""Neutralize links and images in model output before Streamlit renders it as markdown.

Answers can echo text injected through documents. A rendered image makes the browser fetch an
attacker-chosen URL (a data-exfiltration channel); a rendered link invites a click. Here, images are
replaced by their alt text, and links and bare URLs become inert code spans the user can still read.
"""

import re

_IMAGE_INLINE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_IMAGE_REF = re.compile(r"!\[([^\]]*)\]\[[^\]]*\]")
_LINK_INLINE = re.compile(r"\[([^\]]+)\]\(\s*([^)\s]+)[^)]*\)")
_LINK_REF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_REF_DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
_AUTOLINK = re.compile(r"<((?:https?|ftp|mailto):[^>\s]+)>", re.IGNORECASE)
_BARE_URL = re.compile(r"(?<![`\w/])((?:https?|ftp)://[^\s<>`)\]]+|www\.[^\s<>`)\]]+)", re.IGNORECASE)


def neutralize_links(markdown: str) -> str:
    text = _IMAGE_INLINE.sub(r"\1", markdown)
    text = _IMAGE_REF.sub(r"\1", text)
    text = _LINK_INLINE.sub(r"\1 (\2)", text)
    text = _LINK_REF.sub(r"\1", text)
    text = _REF_DEFINITION.sub("", text)
    text = _AUTOLINK.sub(r"\1", text)
    return _BARE_URL.sub(r"`\1`", text)
