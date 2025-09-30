from copy import deepcopy
from os import makedirs, path

import yaml
from paths import FastStableDiffusionPaths

from models.settings import Settings


class AppSettings:
    def __init__(self):
        self.config_path = FastStableDiffusionPaths().get_app_settings_path()
        self._stable_diffsuion_models = []
        self._lcm_lora_models =[]
        self._openvino_lcm_models = []
        self._lcm_models = []
        self._config = None

    @property
    def settings(self):
        return self._config

    @property
    def stable_diffsuion_models(self):
        return self._stable_diffsuion_models

    @property
    def openvino_lcm_models(self):
        return self._openvino_lcm_models

    @property
    def lcm_models(self):
        return self._lcm_models

    @property
    def lcm_lora_models(self):
        return self._lcm_lora_models

    

    def load(self, skip_file=False):
        if skip_file:
            print("Skipping config file")
            settings_dict = self._load_default()
            self._config = Settings.model_validate(settings_dict)
        else:
            if not path.exists(self.config_path):
                base_dir = path.dirname(self.config_path)
                if not path.exists(base_dir):
                    makedirs(base_dir)
                try:
                    print("Settings not found creating default settings")
                    with open(self.config_path, "w") as file:
                        yaml.dump(
                            self._load_default(),
                            file,
                        )
                except Exception as ex:
                    print(f"Error in creating settings : {ex}")
                    exit()
            try:
                with open(self.config_path) as file:
                    settings_dict = yaml.safe_load(file)
                    self._config = Settings.model_validate(settings_dict)
            except Exception as ex:
                print(f"Error in loading settings : {ex}")

    def save(self):
        try:
            with open(self.config_path, "w") as file:
                tmp_cfg = deepcopy(self._config)
                tmp_cfg.lcm_diffusion_setting.init_image = None
                configurations = tmp_cfg.model_dump(
                    exclude=["init_image"],
                )
                if configurations:
                    yaml.dump(configurations, file)
        except Exception as ex:
            print(f"Error in saving settings : {ex}")

    def _load_default(self) -> dict:
        default_config = Settings()
        return default_config.model_dump()
