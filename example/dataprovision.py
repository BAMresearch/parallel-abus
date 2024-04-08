"""Provide data for SHM scenarios and evaluation."""

import pathlib
from typing import Sequence, Union

import pandas as pd
import numpy as np


# data from pure damage states
PURE_DAMAGE_STATES = (
    "ungerissen",
    "gerissen_1",
    "gerissen_2",
    "mit_nut",
)

# data from pure damage states and shm scenarios
SCENARIO_TYPES = (
    "ungerissen",
    "gerissen_1",
    "gerissen_2",
    "mit_nut",
)


THIS_MODULE_FILE = pathlib.Path(__file__)
datapath = THIS_MODULE_FILE.parent / "data"
SYNTH_DATA_FOLDER = THIS_MODULE_FILE.parent / "data/"


df_Difference = pd.read_pickle(
    datapath / "04_Wege_Referenz_difference.pandas_0_24_2.pickle"
)
# convert to normalized units: [m] and [rad]:
df_Difference["W1"] = df_Difference["W1"] / 1000
# convert to [m]
df_Difference["W2"] = df_Difference["W2"] / 1000
# convert to [m]
df_Difference["W3"] = df_Difference["W3"] / 1000
# convert from ° to rad
df_Difference["Neigungssensor_3_k01"] = (
    df_Difference["Neigungssensor_3_k01"] / 180 * np.pi
)
# convert from ° to rad
df_Difference["Neigungssensor_4_k02"] = (
    df_Difference["Neigungssensor_4_k02"] / 180 * np.pi
)

df_static_flat = df_Difference.reset_index([0, 1, 2])
df_static_flat["Position"] = df_static_flat["Position"].cat.remove_unused_categories()
df_static_flat["Schadenstyp"] = df_static_flat[
    "Schadenstyp"
].cat.remove_unused_categories()

# drop data that isn't usefull
# gerissen_2 @ 40°C
# gerissen_1 @ 25°C
# ungerissen @ -25°C

df_static_cleaned = df_static_flat[
    ~(
        (df_static_flat["Schadenstyp"] == "gerissen_2")
        & (df_static_flat["NennTemperatur"] == 40)
    )
    & ~(
        (df_static_flat["Schadenstyp"] == "gerissen_1")
        & (df_static_flat["NennTemperatur"] == 25)
    )
    & ~(
        (df_static_flat["Schadenstyp"] == "ungerissen")
        & (df_static_flat["NennTemperatur"] == -25)
    )
    & ~(
        (df_static_flat["Schadenstyp"] == "mit_nut")
        & (df_static_flat["NennTemperatur"] == 25)
    )
    & ~(
        (df_static_flat["Schadenstyp"] == "mit_nut")
        & (df_static_flat["NennTemperatur"] == 40)
    )
]

df_fdd = pd.read_pickle(
    datapath / "20200218_fdd_aggregation.pandas_0_25_3.pickle"
).sort_values(by=["Schadenstyp", "Temperatur"], axis="index")
df_ssi = pd.read_pickle(
    datapath / "ssi_aggregated_manually.pandas_0_25_3.pickle"
).sort_values(by=["Schadenstyp", "Temperatur"], axis="index")

df_fdd_cleaned = df_fdd[
    ~((df_fdd["Schadenstyp"] == "gerissen_2") & (df_fdd["Temperatur"] == 40))
    & ~((df_fdd["Schadenstyp"] == "mit_nut") & (df_fdd["Temperatur"] == 40))
]

df_ssi_cleaned = df_ssi[
    ~((df_ssi["Schadenstyp"] == "gerissen_2") & (df_ssi["Temperatur"] == 40))
    & ~((df_ssi["Schadenstyp"] == "mit_nut") & (df_ssi["Temperatur"] == 40))
]

df_static_allclean = df_static_flat[
    ~(
        (df_static_flat["Schadenstyp"] == "gerissen_2")
        & (df_static_flat["NennTemperatur"] == 40)
    )
    & ~(
        (df_static_flat["Schadenstyp"] == "gerissen_1")
        & (df_static_flat["NennTemperatur"] == 25)
    )
    & ~(
        (df_static_flat["Schadenstyp"] == "ungerissen")
        & (df_static_flat["NennTemperatur"] == -25)
    )
    & ~(
        (df_static_flat["Schadenstyp"] == "mit_nut")
        & (df_static_flat["NennTemperatur"] == 40)
    )
]

df_fdd_allclean = df_fdd[
    ~((df_fdd["Schadenstyp"] == "gerissen_2") & (df_fdd["Temperatur"] == 40))
    & ~((df_fdd["Schadenstyp"] == "mit_nut") & (df_fdd["Temperatur"] == 40))
    & ~((df_fdd["Schadenstyp"] == "gerissen_1") & (df_fdd["Temperatur"] == 25))
    & ~((df_fdd["Schadenstyp"] == "ungerissen") & (df_fdd["Temperatur"] == -25))
]

df_ssi_allclean = df_ssi[
    ~((df_ssi["Schadenstyp"] == "gerissen_2") & (df_ssi["Temperatur"] == 40))
    & ~((df_ssi["Schadenstyp"] == "mit_nut") & (df_ssi["Temperatur"] == 40))
    & ~((df_ssi["Schadenstyp"] == "gerissen_1") & (df_ssi["Temperatur"] == 25))
    & ~((df_ssi["Schadenstyp"] == "ungerissen") & (df_ssi["Temperatur"] == -25))
]

all_dynamic_damage_types = df_fdd["Schadenstyp"].unique()
all_dynamic_temperatures = df_fdd["Temperatur"].unique()

all_static_damage_types = df_static_cleaned["Schadenstyp"].unique()
all_static_temperatures = df_static_cleaned["NennTemperatur"].unique()


def static_data(
    damage_type: Union[None, str, Sequence[str]] = None,
    temperature: Union[None, int, Sequence[int]] = None,
    clean: bool = True,
    in_english: bool = False,
) -> pd.DataFrame:
    """Reduce the static data to the specified damage_type(s) and temperature(s).

    Args:
        damage_type (Union[None, str, Sequence[str]], optional): If provided, reduce data to the specified damage type(s). Defaults to None.
        temperature (Union[None, int, Sequence[int]], optional): If provided, reduce data to the specified temperature(s). Defaults to None.
        clean (bool, optional): Only use outlier-removed data. Defaults to True.
        in_english (bool, optional): use English names

    Raises:
        TypeError: if 'damage_type' is not a string or a sequence of strings.
        TypeError: if 'temperature' is not an int or a sequence of ints.

    Returns:
        pd.DataFrame: DataFrame containing only the specified experimental results.
    """
    if clean:
        df = df_static_cleaned
    else:
        df = df_static_flat
    if damage_type is None:
        _damage_type = all_static_damage_types
    elif isinstance(damage_type, str):
        assert (
            damage_type in all_static_damage_types
        ), f"damage_type shloud one of: {all_static_damage_types}, but is: {damage_type}"
        _damage_type = [damage_type]
    elif isinstance(damage_type, Sequence):
        _damage_type = damage_type
    else:
        raise TypeError(
            f"damage_type should be str or a sequence, but is: {type(damage_type)}"
        )

    if temperature is None:
        _temperature = all_static_temperatures
    elif isinstance(temperature, int):
        assert temperature in all_static_temperatures
        _temperature = [temperature]
    elif isinstance(temperature, Sequence):
        _temperature = temperature
    else:
        raise TypeError(
            f"temperature should be int or a sequence, but is: {type(temperature)}"
        )
    df = df[
        df["Schadenstyp"].isin(_damage_type) & df["NennTemperatur"].isin(_temperature)
    ]
    with pd.option_context("mode.chained_assignment", None):
        if in_english:
            df["Schadenstyp"] = df["Schadenstyp"].cat.rename_categories(
                {
                    "ungerissen": "undamaged",
                    "gerissen_1": "cracked_1",
                    "gerissen_2": "cracked_2",
                    "mit_nut": "with_notch",
                },
                # inplace=True,
            )

        df["Position"] = df["Position"].cat.remove_unused_categories()
        df["Schadenstyp"] = df["Schadenstyp"].cat.remove_unused_categories()
    return df
