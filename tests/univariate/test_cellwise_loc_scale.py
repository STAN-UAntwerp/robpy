"""The univariate estimators reproduce cellWise::estLocScale and cellWise::wrap.

Reference results: tests/reference/cellwise_reference.R (cellWise 2.5.7).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robpy.univariate import CellwiseOneStepM, HuberOneStepM, OneStepWrapping
from robpy.univariate._cellwise import est_loc_scale, loc_scale_1d, wrap

REFERENCE = Path(__file__).parents[1] / "reference"
DATA = pd.read_csv(REFERENCE / "data" / "univariate.csv").to_numpy()
EXPECTED = pd.read_csv(REFERENCE / "expected" / "loc_scale.csv")


@pytest.mark.parametrize("row", [r for _, r in EXPECTED.iterrows()], ids=lambda r: str(r.name))
def test_loc_scale_matches_cellwise(row):
    loc, scale = loc_scale_1d(DATA[:, row.column - 1], row.type, center=bool(row.center))
    np.testing.assert_allclose(loc, row["loc"], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(scale, row.scale, rtol=1e-8)


@pytest.mark.parametrize(
    "estimator, type",
    [(CellwiseOneStepM, "1stepM"), (HuberOneStepM, "hubhub"), (OneStepWrapping, "wrap")],
)
@pytest.mark.parametrize("center", [True, False])
def test_public_estimators_match_cellwise(estimator, type, center):
    expected = EXPECTED[(EXPECTED.type == type) & (EXPECTED.center == center)]
    for _, row in expected.iterrows():
        x = DATA[:, row.column - 1]
        fitted = estimator(center=center).fit(x[~np.isnan(x)])
        np.testing.assert_allclose(fitted.location, row["loc"], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(fitted.scale, row.scale, rtol=1e-8)


def test_est_loc_scale_handles_columns_with_missing_values():
    loc, scale = est_loc_scale(DATA[:, :3])
    expected = EXPECTED[(EXPECTED.type == "wrap") & EXPECTED.center & (EXPECTED.column <= 3)]
    np.testing.assert_allclose(loc, expected["loc"], rtol=1e-8)
    np.testing.assert_allclose(scale, expected.scale, rtol=1e-8)


def test_wrap_matches_cellwise():
    loc, scale = est_loc_scale(DATA[:, :3])
    expected = pd.read_csv(REFERENCE / "expected" / "wrap.csv").to_numpy()
    np.testing.assert_allclose(wrap(DATA[:, :3], loc, scale), expected, rtol=1e-8)
