import re


def convert_to_snake_case(input_string: str) -> str:
    """
    Convert to snake case.

    :param input_string: Input string that needs to be converted to snake case.
    :return: input_string in snake_case.
    """
    # replace all non-alphanumeric characters (including spaces) with underscores
    input_string_snake_case = re.sub(r"[^a-zA-Z0-9]+", "_", input_string)

    # convert the string to lowercase
    input_string_snake_case = input_string_snake_case.lower()

    # remove leading or trailing underscores
    input_string_snake_case = input_string_snake_case.strip("_")

    # return input_string in snake_case
    return input_string_snake_case
