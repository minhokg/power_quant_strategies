import hashlib
import logging
import os


def shorten_path(path: str, max_n_characters: int = 260) -> str:
    """
    Shorten path.

    On windows the path can by default only be 260 characters long. Thus, try to shorten the filename. If directory path
    is already too long, this will raise a ValueError.
    :param path: full path to the file your try to store
    :param max_n_characters: max number of characters allowed
    :return: potentially shortened full path
    """
    if len(path) <= max_n_characters:
        return path

    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)

    parent_len = len(directory) + 1  # separator

    max_name_len = max_n_characters - parent_len - len(ext)

    if max_name_len <= 0:
        raise ValueError("Directory path too long")

    hash_part = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]  # noqa: S324

    reserved = len(hash_part) + 1
    trimmed_len = max_name_len - reserved
    trimmed_name = name[: max(1, trimmed_len)]

    new_filename = f"{trimmed_name}_{hash_part}{ext}"
    new_path = os.path.join(directory, new_filename)

    logging.debug("Shortened path: %s -> %s", path, new_path)

    return new_path
