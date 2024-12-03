
def find_non_serializable(data):
    """
    Recursively traverse the data structure to find non-serializable objects.
    Args:
        data: The data structure to traverse (dict, list, etc.).
    """
    if isinstance(data, dict):
        for key, value in data.items():
            find_non_serializable(value)
    elif isinstance(data, list):
        for item in data:
            find_non_serializable(item)
    elif not isinstance(data, (str, int, float, bool, type(None), list, dict)):
        print(f"Non-serializable object found: {type(data)} -> {data}")
