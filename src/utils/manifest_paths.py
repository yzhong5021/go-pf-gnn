"""Helpers for resolving manifest path templates."""

from __future__ import annotations

from typing import Optional


def resolve_manifest_path_template(path: Optional[str], *, aspect: Optional[str]) -> Optional[str]:
    """Replace aspect placeholders in manifest paths."""

    if not path or not aspect:
        return path
    aspect_upper = str(aspect).upper()
    aspect_lower = aspect_upper.lower()
    resolved = str(path)
    resolved = resolved.replace("{aspect_lower}", aspect_lower)
    resolved = resolved.replace("{aspect}", aspect_upper)
    return resolved
