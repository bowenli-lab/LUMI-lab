import base64
import io
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

base_url = "http://192.168.1.11:8700"


def get_entries():
    # Placeholder for the method that fetches available entry IDs
    # Replace with the actual implementation
    request_url = base_url + "/entries"
    return requests.get(request_url).json()


def get_entry_readings(entry_id):
    request_url = base_url + f"/entry/{entry_id}/readings"
    response = requests.get(request_url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError("Failed to fetch data. Please check the entry ID.")


def get_entry_gain(entry_id):
    """Get the gain values of the the experiment readings."""
    request_url = base_url + f"/entry/{entry_id}/gain"
    response = requests.get(request_url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError("Failed to fetch data. Please check the entry ID.")


def load_regressor_from_json(file_path=None):
    """Loads the linear regression model from JSON file and returns a simple callable function."""
    if file_path is None:
        cur_file_dir = Path(__file__).parent
        file_path = (
            cur_file_dir
            / "../evaluation/notebooks/empty_control_calib/linear_regressor.json"
        )
    assert os.path.exists(file_path), f"The regressor file does not exist: {file_path}"

    with open(file_path, "r") as json_file:
        regressor_data = json.load(json_file)

    # Return a function that calculates y = mx + b
    def regressor_function(gain_value):
        return regressor_data["slope"] * gain_value + regressor_data["intercept"]

    return regressor_function


def log_and_norm_reading(readings, gain_values=None) -> Dict[str, Dict]:
    """loop and call the _log_and_norm_reading function"""
    if gain_values is None:
        return {key: _log_and_norm_reading(value) for key, value in readings.items()}
    else:
        return {
            key: _log_and_norm_reading(value, gain_values[key])
            for key, value in readings.items()
        }


def _log_and_norm_reading(reading, gain_value=None, clip_negative=True):
    """
    Normalize the readings by calling the log and norm function. Essentially, it logs the readings and normalizes them by the control well readings. The first two wells that have none components are the control wells. The others of none components are the benchmark lipids of MC3. The rest are the experimental wells.

    Args:
        readings (dict): A dictionary containing the readings for each well.
        gain_value (optional, float): The gain value of the experiment. If provided, the empty control well readings will be inferred from the gain value.
        example:
        {
            "A1": {"reading": 0.0, "components": {...}},
            "A2": {"reading": 0.0, "components": {...}},
            ...
        }
        clip_negative (bool): Whether to clip the negative values to zero.

    Returns:
        dict: A dictionary containing the normalized readings for each well.
        example:
        {
            "A1": {"reading": 0.0, "components": {...}, "type": "control"},
            "A2": {"reading": 0.0, "components": {...}, "type": "mc3"},
            ...
        }
    """
    # log transform the data
    log_reading = {}
    for key, value in reading.items():
        log_reading[key] = {
            "reading": np.log2(value["reading"]),
            "components": value["components"],
        }

    # find all wells that have none components
    empty_and_positive_control_wells = []
    for key, value in reading.items():
        if value["components"]["amines"] is None:
            empty_and_positive_control_wells.append(key)
    empty_wells = empty_and_positive_control_wells[:2]
    positive_control_wells = empty_and_positive_control_wells[2:]
    assert empty_wells[0] == "A1"
    assert empty_wells[1] == "B1"

    # assign the type of the well
    for key in log_reading.keys():
        if key in empty_wells:
            log_reading[key]["type"] = "control"
        elif key in positive_control_wells:
            log_reading[key]["type"] = "mc3"
        else:
            log_reading[key]["type"] = "experimental"

    # qc of control wells
    min_control_reading = min([log_reading[key]["reading"] for key in empty_wells])
    if min_control_reading > np.log2(300):
        print(
            f"The control well readings are too high: {np.exp2(min_control_reading):.2f}"
            f" Probably due to light leakage. Autoset control reading to 300."
        )
        min_control_reading = np.log2(300)
    for key in empty_wells:
        if log_reading[key]["reading"] > np.log2(300):
            log_reading[key]["reading"] = min_control_reading

    # normalize the data
    if gain_value is not None:
        # calculate the control well readings from the gain value
        ctrl = load_regressor_from_json()(gain_value)
        print(f"Inferred log control reading from gain {gain_value}: {ctrl:.2f}")
    else:
        ctrl = np.mean([log_reading[key]["reading"] for key in empty_wells])
    for key in log_reading.keys():
        log_reading[key]["reading"] = log_reading[key]["reading"] - ctrl
        if clip_negative:
            log_reading[key]["reading"] = max(log_reading[key]["reading"], 0)

    return log_reading


def qc_entry_readings(nomalized_readings, threshold=3):
    """
    Quality control the readings for each well in the 96-well plate. Each entry can have readings of maximum four replicates. The first two are two repeated readings of one well, and the other two are two repeated readings of another well. Sometimes there are only two replicates, then the third and fourth readings are missing.

    1. check whther the readings are trustworthy: if the difference between the two readings is less than `threshold`, then the readings are trustworthy.
    2. If the readings are trustworthy, then calculate the average of the two readings as the final reading for that well.
    3. Across wells for the same lipid structure, using the larger value as the final value.

    Args:
        readings (dict): A dictionary containing the readings for each well.
        example:
        {
            "0": {
                "A1": {"reading": 0.0, "components": {...}, "type": "control"},
                "A2": {"reading": 0.0, "components": {...}, "type": "mc3"},
                ...
            },
            "1": {
                "A1": {"reading": 0.0, "components": {...}, "type": "control"},
                "A2": {"reading": 0.0, "components": {...}, "type": "mc3"},
                ...
            },
            ...
        }
        threshold (float): The threshold to determine whether the readings are trustworthy. Note this is in the log scale.

    Returns:
        dict: A dictionary containing the normalized readings for each well.
        example:
        {
            "A1": {"reading": 0.0, "components": {...}, "type": "control"},
            "A2": {"reading": 0.0, "components": {...}, "type": "mc3"},
            ...
        }
    """
    qc_data = {}
    for key, readings in nomalized_readings.items():
        for well, reading in readings.items():
            if well not in qc_data:
                qc_data[well] = {
                    "reading": [],
                    "components": reading["components"],
                    "type": reading["type"],
                }
            qc_data[well]["reading"].append(reading["reading"])

    def _process_data(data0, data1, threshold, message_prefix=""):
        if abs(data0 - data1) < threshold:
            return np.mean([data0, data1])
        elif max(data0, data1) > 7:
            return max(data0, data1)
        else:
            print(
                f"{message_prefix} The difference between the two readings ({data0:.2f},{data1:.2f}) is larger than {threshold}."
            )
            return np.mean([data0, data1])

    for well, data in qc_data.items():
        if len(data["reading"]) == 1:
            qc_data[well]["reading"] = data["reading"][0]
        elif len(data["reading"]) == 2:
            data0, data1 = data["reading"]
            qc_data[well]["reading"] = _process_data(
                data0, data1, threshold, f"Replicates of {well}:"
            )
        elif len(data["reading"]) == 4:
            data0, data1, data2, data3 = data["reading"]
            reading1 = _process_data(
                data0, data1, threshold, f"Replicates #0,1 of well {well}:"
            )
            reading2 = _process_data(
                data2, data3, threshold, f"Replicates #2,3 of well {well}:"
            )

            qc_data[well]["reading"] = max(reading1, reading2)
        else:
            qc_data[well]["reading"] = np.nan

    return qc_data


def get_lipid_id2smiles():
    cur_file_dir = Path(__file__).parent
    lipid_library_file = cur_file_dir / "../data_process/220k_library_with_meta.csv"
    assert os.path.exists(lipid_library_file), "The lipid library file does not exist."
    lipid_library = pd.read_csv(lipid_library_file)
    # select columns
    lipid_library = lipid_library[
        [
            "component_str",
            "combined_mol_SMILES",
            "A_name",
            "A_smiles",
            "B_name",
            "B_smiles",
            "C_name",
            "C_smiles",
            "D_name",
            "D_smiles",
        ]
    ]
    lipid_id2smiles = lipid_library.set_index("component_str").to_dict(orient="index")
    return lipid_id2smiles


def data2df(integrated_data):
    # make the integrated data to dataframe, including the experimental type readings
    # each row contains the recordings for a lipid structure of AxBxCxDx
    # each row should be |name|max|mean|std|reading1|reading2|reading3|readingN|
    df_data = {}
    for entry_id, data in integrated_data.items():
        for well, reading in data.items():
            if reading["type"] == "experimental":
                lipid_structure = "".join(
                    [
                        reading["components"][key]
                        for key in sorted(reading["components"].keys())
                    ]
                )
                if lipid_structure not in df_data:
                    df_data[lipid_structure] = {}
                    df_data[lipid_structure]["amine"] = reading["components"]["amines"]
                    df_data[lipid_structure]["isocyanide"] = reading["components"][
                        "isocyanide"
                    ]
                    df_data[lipid_structure]["aldehyde"] = reading["components"][
                        "lipid_aldehyde"
                    ]
                    df_data[lipid_structure]["carboxylic_acid"] = reading["components"][
                        "lipid_carboxylic_acid"
                    ]
                df_data[lipid_structure][f"reading.{entry_id}"] = reading["reading"]
    # make the dataframe
    df = pd.DataFrame(df_data).T
    # add the mean, std, and max columns
    df.insert(4, "max", df.filter(like="reading").max(axis=1, skipna=True))
    df.insert(5, "mean", df.filter(like="reading").mean(axis=1, skipna=True))
    df.insert(6, "std", df.filter(like="reading").std(axis=1, skipna=True))

    # add the similes column by searching the lipid_id2smiles
    lipid_id2smiles = get_lipid_id2smiles()
    smiles = df.index.map(
        lambda x: (
            lipid_id2smiles[x]["combined_mol_SMILES"] if x in lipid_id2smiles else None
        )
    )
    df.insert(0, "smiles", smiles)

    mols = df["smiles"].map(Chem.MolFromSmiles)
    # imgs = mols.map(Draw.MolToImage)  # type of img is PIL.PngImagePlugin.PngImageFile
    # base64_imgs = []
    # for img in imgs:
    #     img_byte_array = io.BytesIO()
    #     img.save(img_byte_array, format="PNG")
    #     img_byte_array = img_byte_array.getvalue()
    #     base64_imgs.append(
    #         "data:image/png;base64," + base64.b64encode(img_byte_array).decode("utf-8")
    #     )
    # imgs = base64_imgs

    df.insert(1, "mol_img", mols)

    # sort the dataframe by the max value
    df = df.sort_values(by="max", ascending=False)
    return df
