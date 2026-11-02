"""The letter targets are legal nodal patterns on either plate orientation.

Every curve must either close on itself or end on the plate boundary, clear of
the clamped corner squares, and stay on the plate.  Junctions are not checked
here -- the letters are constructed so that lines only cross (four branches),
which ``examples/shapes.py`` documents letter by letter.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples import shapes

LETTERS = {
    "T": shapes.letter_t_nodal, "Y": shapes.letter_y_nodal, "Z": shapes.letter_z_nodal,
    "B": shapes.letter_b_nodal, "M": shapes.letter_m_nodal, "L": shapes.letter_l_nodal,
    "W": shapes.letter_w_nodal, "X": shapes.letter_x_nodal, "P": shapes.letter_p_nodal,
}
PLATES = {"upright": (2.0, 2.4), "sideways": (2.4, 2.0)}
CLAMP = 0.2


@pytest.mark.parametrize("plate", PLATES)
@pytest.mark.parametrize("letter", LETTERS)
def test_letter_is_a_legal_nodal_pattern(letter, plate):
    lx, ly = PLATES[plate]
    curves = LETTERS[letter]((lx, ly))
    assert curves, "a letter needs at least one stroke"
    for curve in curves:
        assert curve.ndim == 2 and curve.shape[1] == 2
        assert curve[:, 0].min() >= -1e-9 and curve[:, 0].max() <= lx + 1e-9
        assert curve[:, 1].min() >= -1e-9 and curve[:, 1].max() <= ly + 1e-9
        if np.allclose(curve[0], curve[-1]):
            continue  # a closed loop needs no boundary contact
        for x, y in (curve[0], curve[-1]):
            assert min(x, lx - x, y, ly - y) < 1e-9, f"end ({x:.2f}, {y:.2f}) is off the edge"
            in_clamp = min(x, lx - x) < CLAMP and min(y, ly - y) < CLAMP
            assert not in_clamp, f"end ({x:.2f}, {y:.2f}) is inside a clamped corner"


@pytest.mark.parametrize("letter", LETTERS)
def test_letters_are_smooth_except_where_strokes_cross(letter):
    """Fillets remove the drawing's sharp corners: no turn sharper than 90 deg."""
    for curve in LETTERS[letter](PLATES["upright"]):
        seg = np.diff(curve, axis=0)
        seg = seg[np.linalg.norm(seg, axis=1) > 1e-12]
        if len(seg) < 2:
            continue  # a single straight stroke has no turns
        heading = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
        turn = np.abs((np.diff(heading) + 180.0) % 360.0 - 180.0)
        assert turn.max() <= 90.0 + 1e-6
