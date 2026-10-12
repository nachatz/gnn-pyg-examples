import warnings


def preload() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.script.*")
