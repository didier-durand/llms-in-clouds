import os
from pathlib import Path

from utils import get_project_root

DEBUG: bool = True
DEV_TEAM = ["didduran"]

ROOT_PATH = Path(get_project_root())
CFN_PATH = Path(os.path.join(ROOT_PATH, "cfn"))
DATA_PATH = Path(os.path.join(ROOT_PATH, "data"))
DOCKER_PATH = Path(os.path.join(ROOT_PATH, "docker"))
EXTEND_PATH = Path(os.path.join(ROOT_PATH, "extend"))
SETUP_PATCH = Path(os.path.join(ROOT_PATH, "start_sglang.sh"))
