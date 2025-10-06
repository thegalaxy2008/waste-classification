def get_cwd():
    """Get the current working directory.

    Returns:
        str: The current working directory.
    """
    
    from pathlib import Path
    return (Path(__file__).parent.parent)
