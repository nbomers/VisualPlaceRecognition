"""Geraetewahl an einer Stelle, fuer 04, 05, locate.py und die Experimente."""


def pick_device(wunsch=None):
    """cuda > mps > cpu, oder der ausdruecklich gewuenschte Name."""
    import torch

    if wunsch:
        return str(wunsch)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
