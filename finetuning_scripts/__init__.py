from __future__ import annotations


def adjust_path_for_scripts():
    import sys
    from pathlib import Path
    finetune_dir = str(Path(__file__).parent.parent.resolve().absolute())
    if finetune_dir not in sys.path:
        sys.path.insert(0, finetune_dir)


# Using this by default right now as init files are always executed.
# Probably not a very good idea...
adjust_path_for_scripts()
