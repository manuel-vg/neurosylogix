import os
import json

def open_json(json_path):
    '''
    input: a path to a json file
    output: the json file or None (if invalid)
    '''
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None

def save_json(json_path, data):
    '''
    input: JSON path (with optional directory part) and JSON actual data
    output: writes a JSON file
    '''
    directory = os.path.dirname(json_path)

    # if path includes dirs
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'File successfully saved to "{json_path}"')