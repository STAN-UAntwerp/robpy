import robpy.datasets

import numpy as np


def test_can_load_all_datasets():
    datasets = ["telephone", "stars", "animals", "topgear", "glass"]
    for dataset in datasets:
        data = getattr(robpy.datasets, f"load_{dataset}")().data
        assert isinstance(data, np.ndarray), f"dataset {dataset} failed: {type(data.data)}"
