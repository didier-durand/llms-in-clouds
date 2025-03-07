import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import boto3


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs.items():
            setattr(func, k, kwargs[k]) # noqa
        return func

    return decorate


def get_project_root() -> Path:
    root_path = Path(__file__).parent.parent
    print("root path: ", root_path)
    return root_path


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_current_account() -> str:
    return boto3.client('sts').get_caller_identity().get('Account')


def get_current_region() -> str:
    return boto3.session.Session().region_name


def check_response(resp: dict, http_status_code=200) -> bool:
    assert (resp['ResponseMetadata']['HTTPStatusCode']
            == http_status_code), ("unexpected HTTP status code: "
                                   + str(http_status_code)
                                   + " <> " +
                                   str(resp['ResponseMetadata']['HTTPStatusCode']))
    return True


def to_json(data: dict | list = None, indent: int = 4) -> str:
    return json.dumps(data, indent=indent, default=str)


def on_dev_machine() -> bool:
    return "didduran" in os.environ["HOME"]


def read_file(file_path: Path = None, throw: bool = True) -> str | Exception:
    assert file_path is not None
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except Exception as exception:  # noqa pylint: disable=W0718
        if not throw:
            print("read_file: " + str(file_path)
                  + " - exception: " + str(type(exception))
                  + " - msg:" + str(exception))
            return exception
        raise exception
    return content


def exec_os_command(command: list[str] | str = None) -> tuple[Exception | None, int | None, str | None, str | None]:
    if isinstance(command, str):
        command = command.split(" ")
    print("executing:", " ".join(command))
    try:
        process: subprocess.CompletedProcess = subprocess.run(command,
                                                              capture_output=True,
                                                              text=True, check=False)
    except Exception as exception:  # noqa pylint: disable=W0718
        return exception, None, None, None
    return None, process.returncode, process.stdout, process.stderr


def remove_directory_tree(start_directory: Path):
    """Recursively and permanently removes the specified directory, all of its
    subdirectories, and every file contained in any of those folders."""
    for path in start_directory.iterdir():
        if path.is_file():
            path.unlink()
        else:
            remove_directory_tree(path)
    start_directory.rmdir()
