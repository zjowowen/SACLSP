def merge_dicts(dict1, dict2):
    for key, value in dict1.items():
        if key in dict2 and isinstance(value, dict) and isinstance(dict2[key], dict):
            # Both values are dictionaries, so merge them recursively
            merge_dicts(value, dict2[key])
        else:
            # Either the key doesn't exist in dict2 or the values are not dictionaries
            dict2[key] = value
    
    return dict2
