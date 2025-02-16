import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sdl_orchestration.logger import logger


class LiquidSamplerCalibration:
    """
    This class is used to calibrate the liquid sampler.
    """
    def __init__(
        self,
        file_path: str,
        cache_path: str = None,
        use_cache: bool = True,
    ):
        """
        Args:
            file_path: The path to the csv file containing the calibration data.
            cache_path: The path to the csv file to save the calibration data to.
            use_cache: Whether to load the cache file if it exists.
        """
        self.data = pd.read_csv(file_path)
        # get the unit values from the header
        self.unit_values = np.array([x for x in self.data.columns[1:]])
        self.data = self.data.replace("x", 100)
        # # fill any cell in the pump models df that is "x"
        # # TODO: fix this

        if use_cache:
            self.pump_models_df = pd.read_csv(cache_path)
        else:
            self.pump_models_df = self._fit_models()
            if cache_path:
                self.pump_models_df.to_csv(cache_path, index=False)

    def _fit_models(self):
        # logger.info("Fitting pump models...")
        pump_models = []
        for index, row in self.data.iterrows():
            volume_values = np.array(row[1:])
            volume_values = volume_values.reshape(-1, 1).astype(np.float64)
            model = LinearRegression()
            model.fit(self.unit_values.reshape(-1, 1).astype(np.float64), volume_values)
            slope = model.coef_[0][0]
            intercept = model.intercept_[0]
            pump_models.append((row.iloc[0], slope, intercept))
        # logger.info("Pump models fitted")
        return pd.DataFrame(pump_models, columns=['Well', 'Slope', 'Intercept'])

    def predict_volumes(self, unit_values):
        predictions = []
        for index, row in self.pump_models_df.iterrows():
            well = row['Well']
            slope = row['Slope']
            intercept = row['Intercept']
            for unit in unit_values:
                volume = slope * unit + intercept
                predictions.append((well, unit, volume))
        return pd.DataFrame(predictions,
                            columns=['Well', 'Unit', 'Predicted Volume'])

    def calculate_units_for_volume(self, target_volume):
        units_needed = []
        for index, row in self.pump_models_df.iterrows():
            well = row['Well']
            slope = row['Slope']
            intercept = row['Intercept']
            units = (target_volume - intercept) / slope
            units_needed.append((well, units))
        return pd.DataFrame(units_needed, columns=['Well', 'Units Needed'])


if __name__ == "__main__":
    calibration = LiquidSamplerCalibration("/Users/pangkuan/dev/SDL-LNP/control_panel/test/liquid-sampler-calibration.csv",
                                           cache_path="/Users/pangkuan/dev/SDL-LNP/control_panel/test/liquid-sampler-calibration-cache.csv")
