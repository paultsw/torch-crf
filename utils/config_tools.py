"""
config_tools.py: load hyperparameters to/from a JSON.
"""
import json

def json_to_config(json_path):
    """
    Load training/eval parameters from JSON file.
    """
    try:
        with open(json_path, 'r') as jf:
            cfg = json.load(jf)
        return cfg
    except:
        raise Exception("ERR: Could not load config from JSON at: {}".format(json_path))



def config_to_json(cfg, json_path):
    """
    Serialize a configuration to JSON.
    """
    try:
        with open(json_path, 'w') as jf:
            json.dump(cfg, jf, indent=4, sort_keys=True)
    except:
        raise Exception("ERR: Could not write config to JSON at: {}".format(json_path))
