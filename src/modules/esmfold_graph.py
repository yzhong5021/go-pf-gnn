"""Backward-compatible shim for structure_graph."""

from .structure_graph import StructureGraphBuilder as ESMFoldGraphBuilder

__all__ = ["ESMFoldGraphBuilder"]
