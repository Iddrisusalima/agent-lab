"""Agent Lab: a command-line research agent over your own notes and documents.

This package deliberately imports nothing at module level beyond the version
string, so that every submodule (``config``, ``model_client``, ``cost``, the
tools, the retrieval layer, the eval runner) can be imported on its own without
pulling in the CLI entry point (Req 1.1).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
