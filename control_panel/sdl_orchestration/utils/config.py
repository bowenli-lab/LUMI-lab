import os
from typing import Any
import pandas as pd
import toml

from sdl_orchestration import logger

SDL_CONFIG_VAR = "SDL_CONFIG"
REAGENT_CONFIG_VAR = "REAGENT_CONFIG"
SLACK_BOT_TOKEN_VAR = "SLACK_BOT_TOKEN"


class SDL_Config:
    """
    This class is used to retrieve the configuration of the SDL.

    Assume is config file is in toml and environment variables (`SDL_CONFIG`)
    is set to the config file path.
    """

    def __init__(self):
        self.config = None
        self.config_file_path = os.getenv(SDL_CONFIG_VAR)
        self.reagent_config_path = os.getenv(REAGENT_CONFIG_VAR)
        self.slack_bot_token = os.getenv(SLACK_BOT_TOKEN_VAR)
        if self.config_file_path is None:
            raise Exception(f"Environment variable " f"{SDL_CONFIG_VAR} is not set.")
        if self.reagent_config_path is None:
            raise Exception(
                f"Environment variable " f"{REAGENT_CONFIG_VAR} is not set."
            )
        if self.slack_bot_token is None:
            raise Exception(
                f"Environment variable " f"{SLACK_BOT_TOKEN_VAR} is not set."
            )

        with open(self.config_file_path, "r") as f:
            self.config = toml.load(f)

        logger.info(f"SDL config loaded from {self.config_file_path}")
        logger.info(f"SDL config: {self.config}")
        logger.info(f"Reagent config loaded from {self.reagent_config_path}")

        self._parse_config()

    def get(self, key) -> Any:
        return self.config[key]

    def _parse_config(self):
        """
        This method parse all the configuration from the config file.
        """
        self._parse_db_config()
        self._parse_protocol_config()
        self._parse_robot_config()
        self._parse_reagent_config()
        self._parse_opentron_config()
        self._parse_plate_reader_config()
        self._parse_incubator_config()
        self._parse_liquid_sampler_config()
        self._parse_feeder_config()
        self._parse_clamp_config()
        self._parse_notification_config()

    def _parse_clamp_config(self):
        """
        This method parse the clamp configuration from the config file.
        """
        self.clamp_configs = self.config["clamp"]

    def _parse_protocol_config(self):
        """
        This method parse the protocol configuration from the config file.
        """
        self.protocol_configs = self.config["protocol_persistence"]
        if "parsed_protocol_path" in self.protocol_configs:
            # if there is no path, create the path for the parsed protocol dir
            if not os.path.exists(self.protocol_configs["parsed_protocol_path"]):
                os.makedirs(self.protocol_configs["parsed_protocol_path"])

    def _parse_db_config(self) -> None:
        """
        This method parse the database configuration from the config file.
        """
        # get the atlas uri from env var first
        self.atlas_url = os.getenv("ATLAS_URL", None)
        if self.atlas_url is None:
            # if not found in env var, get from config file
            self.atlas_url = (
                self.config["database"]["atlas_url"]
                if "atlas_url" in self.config["database"]
                else None
            )
        if self.atlas_url is None:
            raise Exception(
                "Atlas URI not found in environment variables or " "config file."
            )
        self.database_name = self.config["database"]["database_name"]

    def _parse_robot_config(self) -> None:
        """
        This method parse the robot configuration from the config file.
        """
        self.robot_ip = self.config["robot"]["host"]
        self.robot_port = self.config["robot"]["port"]
        self.robot_time_limit = self.config["robot"]["time_limit"]

    def _parse_reagent_config(self) -> None:
        """
        This method parse the reagent configuration from the config file.
        """
        self.reagent_config = pd.read_csv(self.reagent_config_path)

        self.reagent2motor_mapping = {
            row["Reagent"]: row["Motor_ID"]
            for idx, row in self.reagent_config.iterrows()
        }
        self.motor2reagent_mapping = {
            row["Motor_ID"]: row["Reagent"]
            for idx, row in self.reagent_config.iterrows()
        }
        self.well2motor_mapping = {
            row["Well_cor"]: row["Motor_ID"]
            for idx, row in self.reagent_config.iterrows()
        }
        self.motor2well_mapping = {
            row["Motor_ID"]: row["Well_cor"]
            for idx, row in self.reagent_config.iterrows()
        }

    def _parse_opentron_config(self) -> None:
        """
        This method parse the opentron configuration from the config file.
        Since we will be using two opentrons, we will have two configurations.
        """
        self.opentron_config = {
            0: self.config["opentron0"],
            1: self.config["opentron1"],
        }
        for key, value in self.opentron_config.items():
            # add some common configs
            value["parsed_protocol_path"] = self.protocol_configs[
                "parsed_protocol_path"
            ]
            value["program_path"] = self.protocol_configs["program_path"]
            if (
                self.protocol_configs["parsed_protocol_path"]
                not in value["program_path"]
            ):
                # add the search domain for the parsed protocol path
                value["program_path"].append(
                    self.protocol_configs["parsed_protocol_path"]
                )
            value["labware_path"] = self.protocol_configs["labware_path"]

    def _parse_plate_reader_config(self) -> None:
        """
        This method parse the plate reader configuration from the config file.
        """
        self.plate_reader_ip = self.config["plate_reader"]["host"]
        self.plate_reader_port = self.config["plate_reader"]["port"]
        self.protocol_path = self.config["plate_reader"]["protocol_path"]
        self.reading_output_base = self.config["plate_reader"]["reading_output_path"]
        self.experimental_output_base = self.config["plate_reader"][
            "experiment_output_path"
        ]

    def _parse_incubator_config(self) -> None:
        """
        This method parse the incubator configuration from the config file.
        """
        self.incubator_wait_time_hrs = self.config["incubator"]["wait_time"]
        self.incubator_wait_time_secs = self.incubator_wait_time_hrs * 60 * 60
        self.incubator_ip = self.config["incubator"]["host"]
        self.incubator_port = self.config["incubator"]["port"]

    def _parse_liquid_sampler_config(self) -> None:
        """
        This method parse the liquid sampler configuration from the config file.
        """
        self.liquid_sampler_ip = self.config["liquid_sampler"]["host"]
        self.liquid_sampler_port = self.config["liquid_sampler"]["port"]
        self.liquid_sampler_calibration_file = self.config["liquid_sampler"][
            "calibration_file"
        ]
        self.liquid_sampler_sample_coefficient = self.config["liquid_sampler"][
            "sample_coefficient"
        ]
        self.liquid_sampler_sample_volume = (self.config)["liquid_sampler"][
            "sample_volume"
        ]
        self.liquid_sampler_safe_volume = self.config["liquid_sampler"]["safe_volume"]
        self.liquid_sampler_safe_cap = self.config["liquid_sampler"]["safe_cap"]
        self.liquid_sampler_calibration_cache_file = self.config["liquid_sampler"][
            "calibration_cache_file"
        ]

    def _parse_feeder_config(self) -> None:
        """
        This method parse the feeder configuration from the config file.
        """
        self.feeder_config = {}
        # the name in toml is like feeder0, feeder1, feeder2, feeder3
        # parse and get the integers after feeder to get the device code
        for key, value in self.config.items():
            if key.startswith("feeder"):
                device_code = int(key.replace("feeder", ""))
                self.feeder_config[device_code] = value
                assert "host" in value, (
                    f"`host` not found in feeder config {value} in "
                    f"{self.config_file_path}"
                )
                assert "port" in value, (
                    f"`port` not found in feeder config {value} in"
                    f" {self.config_file_path}"
                )

    def _parse_notification_config(self) -> None:
        """
        This method parse the notification configuration from the config file.
        """
        self.notification_config = self.config["notification"]


sdl_config = SDL_Config()
