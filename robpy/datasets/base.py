import pathlib
import pandas as pd
import numpy as np
from sklearn.utils import Bunch

DATA_FOLDER = pathlib.Path(__file__).parent / "data"
DESCR_FOLDER = pathlib.Path(__file__).parent / "descr"


def _load_data_and_descr(
    data_file_name: str, descr_file_name: str, as_frame: bool, feature_names: list[str]
) -> tuple[pd.DataFrame | np.ndarray, str]:
    """Helper function to load data and description files.

    Args:
        data_file_name (str): filename of the .csv file (must end in .csv)
        descr_file_name (str): filename of the .rst file (must end in .rst)
        as_frame (bool): whether data should be stored as a pandas DataFrame.
            If False, it will be stored as a numpy array.
        feature_names (list[str]): Columns to be selected, all other columns are ignored

    Returns:
        tuple[pd.DataFrame | np.ndarray, str]: The data matrix/dataframe and the description string
    """

    df = pd.read_csv(DATA_FOLDER / data_file_name)

    with open(DESCR_FOLDER / descr_file_name, "r") as f:
        fdescr = f.read()

    data = df[feature_names]

    if not as_frame:
        data = data.values

    return data, fdescr


def load_telephone(*, as_frame=False):
    """Load and return the telephone dataset (regression with outliers).

    The telephone dataset is well-known univariate regression problem with outliers.

    =================   ==============
    Samples                         24
    Dimensionality                   2
    Features            real, positive
    =================   ==============

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.


    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (24, 2)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        feature_names: list
            The names of the dataset columns.
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.

    Examples
    --------
    Fitting a robust regression

    >>> from robpy.datasets import load_telephone
    >>> from robpy.regression import MMEstimator
    >>> data = load_telephone()
    >>> mm = MMEstimator().fit(data.data[:, 0], data.data[:, 1])
    """
    data_file_name = "telephone.csv"
    descr_file_name = "telephone.rst"

    feature_names = ["Year", "Calls"]
    data, fdescr = _load_data_and_descr(
        data_file_name, descr_file_name, as_frame=as_frame, feature_names=feature_names
    )

    return Bunch(
        data=data,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_folder=DATA_FOLDER,
    )


def load_stars(*, as_frame=False):
    """Load and return the Hertzsprung-Russell Diagram Data of Star Cluster CYG OB1
    (covariance/regression).

    The stars dataset is well-known bivariate dataset used for demonstrating
    robust covariance and regression estimators.

    =================   ==============
    Samples                         47
    Dimensionality                   2
    Features            real, positive
    =================   ==============

    Parameters
    ----------

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric).


    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (47, 2)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        feature_names: list
            The names of the dataset columns.
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.


    Examples
    --------
    Fitting a robust covariance estimator

    >>> from robpy.datasets import load_stars
    >>> from robpy.covariance import FastMCDEstimator
    >>> data = load_stars()
    >>> mcd = FastMCDEstimator().fit(data.data)
    """
    data_file_name = "stars.csv"
    descr_file_name = "stars.rst"

    feature_names = ["Te", "light"]

    data, fdescr = _load_data_and_descr(data_file_name, descr_file_name, as_frame, feature_names)

    return Bunch(
        data=data,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_folder=DATA_FOLDER,
    )


def load_animals(*, as_frame=False):
    """Load and return the Animals2 dataset from robustbase (R) (covariance / regression).

    The animals dataset is a bivariate dataset used for demonstrating robust covariance estimators.

    =================   ==============
    Samples                         65
    Dimensionality                   2
    Features            real, positive
    =================   ==============

    Parameters
    ----------

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric).


    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (65, 2)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        feature_names: list
            The names of the dataset columns.
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.


    Examples
    --------
    Fitting a robust covariance estimator

    >>> from robpy.datasets import load_animals
    >>> from robpy.covariance import FastMCDEstimator
    >>> data = load_animals()
    >>> mcd = FastMCDEstimator().fit(data.data)
    """
    data_file_name = "animals.csv"
    descr_file_name = "animals.rst"

    feature_names = ["body", "brain"]

    data, fdescr = _load_data_and_descr(data_file_name, descr_file_name, as_frame, feature_names)

    return Bunch(
        data=data,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_folder=DATA_FOLDER,
    )


def load_topgear(*, as_frame=False):
    """Load and return the TopGear dataset from robustHD (R) (regression).

    The TopGear dataset is a mixed variable dataset used for demonstrating
    robust regression estimators.

    =================   ==============
    Samples                        297
    Dimensionality                  32
    Features                     mixed
    =================   ==============

    Parameters
    ----------

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric).


    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (297, 32)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        feature_names: list
            The names of the dataset columns.
        categorical_features: list
            The names of the categorical features.
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.


    Examples
    --------
    Fitting a robust regression estimator

    >>> from robpy.datasets import load_topgear
    >>> from robpy.regression import FastLTSRegressor
    >>> data = load_topgear(as_frame=True)
    >>> data.data = data.data.dropna(subset=["Cylinders", "Torque", "TopSpeed", "Price"])
    >>> lts = FastLTSRegressor().fit(
            data.data[["Cylinders", "Torque", "TopSpeed"]], data.data["Price"]
        )
    """
    data_file_name = "topgear.csv"
    desc_file_name = "topgear.rst"

    feature_names = [
        "Maker",
        "Model",
        "Type",
        "Fuel",
        "Price",
        "Cylinders",
        "Displacement",
        "DriveWheel",
        "BHP",
        "Torque",
        "Acceleration",
        "TopSpeed",
        "MPG",
        "Weight",
        "Length",
        "Width",
        "Height",
        "AdaptiveHeadlights",
        "AdjustableSteering",
        "AlarmSystem",
        "Automatic",
        "Bluetooth",
        "ClimateControl",
        "CruiseControl",
        "ElectricSeats",
        "Leather",
        "ParkingSensors",
        "PowerSteering",
        "SatNav",
        "ESP",
        "Verdict",
        "Origin",
    ]
    categorical_features = [
        "Maker",
        "Model",
        "Type",
        "Fuel",
        "DriveWheel",
        "AdaptiveHeadlights",
        "AdjustableSteering",
        "AlarmSystem",
        "Automatic",
        "Bluetooth",
        "ClimateControl",
        "CruiseControl",
        "ElectricSeats",
        "Leather",
        "ParkingSensors",
        "PowerSteering",
        "SatNav",
        "ESP",
        "Origin",
    ]

    data, fdescr = _load_data_and_descr(data_file_name, desc_file_name, as_frame, feature_names)

    return Bunch(
        data=data,
        DESCR=fdescr,
        feature_names=feature_names,
        categorical_features=categorical_features,
        filename=data_file_name,
        data_folder=DATA_FOLDER,
    )


def load_glass(*, as_frame=False):
    """Load and return the glass dataset from cellWise (R) (outlier detection).

    The glass dataset is a high dimensional dataset used for demonstrating outlier detectors.

    =================   ==============
    Samples                        180
    Dimensionality                 750
    Features            real, positive
    =================   ==============

    Parameters
    ----------

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric).


    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (65, 2)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        feature_names: list
            The names of the dataset columns.
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.
    """
    data_file_name = "glass.csv"
    descr_file_name = "glass.rst"

    feature_names = [f"V{i}" for i in range(1, 751)]

    data, fdescr = _load_data_and_descr(data_file_name, descr_file_name, as_frame, feature_names)

    return Bunch(
        data=data,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_folder=DATA_FOLDER,
    )
