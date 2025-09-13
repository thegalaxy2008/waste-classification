def get_cwd():
    from pathlib import Path
    return str(Path(__file__).parent.parent)
