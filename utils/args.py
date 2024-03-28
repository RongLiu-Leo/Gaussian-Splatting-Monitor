import argparse
import yaml
import json

class ConfigLoader:
    def __init__(self, yaml_file_path=None):
        if yaml_file_path:
            self.cfg = self.load_yaml(yaml_file_path)
        else:
            self.cfg = {}

    def load_yaml(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            return yaml.safe_load(file)

    def parse_value(self, value):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def update_dict(self, d, key_path, value):
        keys = key_path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = self.parse_value(value)

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Update a dictionary using command line arguments.')
        parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Path to the YAML configuration file')
        parser.add_argument('updates', nargs='*', help='Updates in the format key.path=value')
        args = parser.parse_args()

        if args.config:
            self.cfg = self.load_yaml(args.config)
            self.cfg['config_path'] = args.config

        for update in args.updates:
            key_path, value = update.split('=', 1)
            self.update_dict(self.cfg, key_path, value)

if __name__ == "__main__":
    config_loader = ConfigLoader()
    config_loader.parse_args()
    print(config_loader.cfg)
