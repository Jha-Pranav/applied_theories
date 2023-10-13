import os
import shutil
from glob import glob


def delete_files_and_dirs(path):
    files_to_delete = [
        ".vscode",
        "**/*.pyc",
        ".Python",
        ".DS_Store",
        "__pycache__",
        "*-checkpoint.ipynb",
        ".ipynb_checkpoints",
        "**.ckpt"
    ]

    for root, dirs, files in os.walk(path):
        for name in files_to_delete:
            for f in glob(os.path.join(root, name), recursive=True):
                if os.path.isfile(f):
                    os.remove(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)


# if __name__ == '__main__':
delete_files_and_dirs(".")
