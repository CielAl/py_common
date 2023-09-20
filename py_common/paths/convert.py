"""
[Driver_Letter]:/path/to/file <---> /mnt/[lower driver letter]/path/to/file
"""
import os
import platform


def windows_to_linux(win_path: str, prefix: str = "/mnt") -> str:
    normalized_win_path = win_path.replace("\\", "/")
    drive, path_without_drive = normalized_win_path.split(":")
    linux_path = "/".join([prefix, drive.lower(), path_without_drive.lstrip("/")])
    return linux_path


def linux_to_windows(linux_path: str) -> str:
    parts = linux_path.split("/")
    drive = parts[2].upper()
    new_parts = [f"{drive.upper()}:"] + parts[3:]
    win_path = os.sep.join(new_parts)
    return win_path


def convert_path(path: str) -> str:
    current_os = platform.system()

    # If we're on Linux
    if current_os == "Linux":
        # Check if the path is in Windows format
        if "\\" in path or ":" in path:
            return windows_to_linux(path)
        return path

    # If we're on Windows
    elif current_os == "Windows":
        # Check if the path is in Linux format with /mnt/
        if path.startswith("/mnt/"):
            return linux_to_windows(path)
        return path

    # If we're on a different OS (e.g., Mac)
    else:
        raise ValueError(f"OS {current_os} not supported")