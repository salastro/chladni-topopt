"""Target Chladni patterns as plain arrays of 2-D points.

Deliberately unabstracted: every function here just returns a list of ``(N, 2)``
numpy arrays in plate coordinates (metres).  Hand-drawn point lists, SVG path
samples or anything else in that form work just as well -- pass them straight
to :func:`chladni_topopt.selection.build_selection_vectors`.

Most shapes here are closed loops (``loops=``).  :func:`letter_s` is an open
curve (``open_curves=``): a nodal line has to close on itself or end on the
plate boundary, so an open target runs all the way to the edges.
"""

from __future__ import annotations

import numpy as np


def letter_d(
    cx: float,
    cy: float,
    width: float,
    height: float,
    n: int = 240,
) -> list[np.ndarray]:
    """Outline of the letter D, centred on ``(cx, cy)``.

    A straight left stem closed by an elliptical right bulge -- the target used
    for Case 1 of the paper (Fig. 4).
    """
    hw, hh = 0.5 * width, 0.5 * height
    x0, x1 = cx - hw, cx + hw
    y0, y1 = cy - hh, cy + hh

    n_arc = max(n - n // 4, 8)
    n_stem = max(n // 4, 4)

    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_arc)
    arc = np.column_stack([x0 + (x1 - x0) * np.cos(theta), cy + hh * np.sin(theta)])
    stem = np.column_stack([np.full(n_stem, x0), np.linspace(y1, y0, n_stem)])

    # Arc runs bottom -> top on the right; stem returns top -> bottom.
    return [np.vstack([arc, stem[1:-1]])]


def letter_t(
    cx: float,
    cy: float,
    width: float,
    height: float,
    bar: float = 0.28,
    stem: float = 0.34,
) -> list[np.ndarray]:
    """Outline of the letter T.  ``bar`` and ``stem`` are fractions of height/width."""
    hw, hh = 0.5 * width, 0.5 * height
    bar_h = bar * height
    stem_w = stem * width
    x0, x1 = cx - hw, cx + hw
    y0, y1 = cy - hh, cy + hh
    sx0, sx1 = cx - 0.5 * stem_w, cx + 0.5 * stem_w
    ybar = y1 - bar_h

    return [
        np.array(
            [
                [x0, y1],
                [x1, y1],
                [x1, ybar],
                [sx1, ybar],
                [sx1, y0],
                [sx0, y0],
                [sx0, ybar],
                [x0, ybar],
            ]
        )
    ]


def letter_u(
    cx: float,
    cy: float,
    width: float,
    height: float,
    thickness: float = 0.3,
    n: int = 120,
) -> list[np.ndarray]:
    """Outline of the letter U.  ``thickness`` is a fraction of the width."""
    hw, hh = 0.5 * width, 0.5 * height
    t = thickness * width
    x0, x1 = cx - hw, cx + hw
    y1 = cy + hh
    bowl_top = cy - hh + (hw - t)

    theta = np.linspace(np.pi, 2.0 * np.pi, n)
    outer = np.column_stack([cx + hw * np.cos(theta), bowl_top + hw * np.sin(theta)])
    inner = np.column_stack(
        [cx + (hw - t) * np.cos(theta), bowl_top + (hw - t) * np.sin(theta)]
    )

    return [
        np.vstack(
            [
                [[x0, y1]],
                outer,
                [[x1, y1], [x1 - t, y1]],
                inner[::-1],
                [[x0 + t, y1]],
            ]
        )
    ]


def circle(cx: float, cy: float, radius: float, n: int = 180) -> list[np.ndarray]:
    """A circle -- the simplest target, handy for tests and sanity checks."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return [np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])]


def _ray_to_boundary(
    start: np.ndarray, direction: np.ndarray, lx: float, ly: float, spacing: float
) -> np.ndarray:
    """Points from ``start`` along ``direction`` to the edge of ``[0,lx]x[0,ly]``."""
    direction = direction / np.linalg.norm(direction)
    hits = []
    for axis, limit in ((0, 0.0), (0, lx), (1, 0.0), (1, ly)):
        if direction[axis] != 0.0:
            t = (limit - start[axis]) / direction[axis]
            if t > 1e-12:
                hits.append(t)
    t_end = min(hits)
    n = max(int(np.ceil(t_end / spacing)), 1)
    t = np.linspace(0.0, t_end, n + 1)[1:]
    return start + t[:, None] * direction


def letter_s(
    cx: float,
    cy: float,
    radius: float,
    plate: tuple[float, float],
    hook: float = 70.0,
    n: int = 240,
) -> list[np.ndarray]:
    """An S running from the left edge of the plate to the right edge.

    Returns one **open** curve -- pass it as ``open_curves``, not ``loops``.

    The S is two arcs of ``radius`` meeting at ``(cx, cy)``: the upper one
    leaves the middle heading left, climbs round the top and stops ``hook``
    degrees short of pointing straight down.  From there the curve carries
    on along its own tangent to the plate boundary.  The lower half is the
    point reflection of the upper half through ``(cx, cy)``, so the two ends
    reach opposite edges symmetrically.  Extending along the tangent keeps
    the curve smooth -- no cusp where the arc meets the tail.

    ``hook`` = 90 gives horizontal tails; smaller values sweep the ends
    downwards (top) and upwards (bottom).  Check that the ends land clear of
    any clamped region.
    """
    lx, ly = plate
    centre = np.array([cx, cy])
    upper_centre = centre + np.array([0.0, radius])

    # Upper arc: from the middle (270 deg) clockwise round to the hook angle.
    n_arc = max(n // 3, 16)
    theta = np.radians(np.linspace(270.0, hook, n_arc))
    arc = upper_centre + radius * np.column_stack([np.cos(theta), np.sin(theta)])
    # Moving clockwise, the tangent at angle th is (sin th, -cos th).
    end = theta[-1]
    tangent = np.array([np.sin(end), -np.cos(end)])
    spacing = np.linalg.norm(arc[1] - arc[0])
    tail = _ray_to_boundary(arc[-1], tangent, lx, ly, spacing)

    upper = np.vstack([arc, tail])  # from the middle outward
    lower = 2.0 * centre - upper  # its point reflection, also middle outward
    return [np.vstack([lower[::-1], upper[1:]])]


def u_curve(
    cx: float,
    width: float,
    bottom: float,
    plate: tuple[float, float],
    n: int = 240,
) -> list[np.ndarray]:
    """A U whose arms run straight up to the top edge of the plate.

    Returns one **open** curve -- pass it as ``open_curves``, not ``loops``.
    (:func:`letter_u` is a different thing: the closed outline of a thick
    letter U.)

    Two vertical arms ``width`` apart, centred on ``cx``, joined by a
    semicircle whose lowest point is at height ``bottom``.  The arms are
    tangent to the semicircle, so the curve is smooth, and they meet the top
    edge at right angles.  Keep the arms clear of any clamped corners.
    """
    _, ly = plate
    r = 0.5 * width
    centre_y = bottom + r
    n_arc = max(n // 3, 16)
    theta = np.linspace(np.pi, 2.0 * np.pi, n_arc)  # left, round the bottom, right
    arc = np.column_stack([cx + r * np.cos(theta), centre_y + r * np.sin(theta)])
    spacing = np.linalg.norm(arc[1] - arc[0])
    n_arm = max(int(np.ceil((ly - centre_y) / spacing)), 1)
    arm_y = np.linspace(ly, centre_y, n_arm + 1)[:-1]  # top edge down to the arc
    left = np.column_stack([np.full_like(arm_y, cx - r), arm_y])
    right = np.column_stack([np.full_like(arm_y, cx + r), arm_y[::-1]])
    return [np.vstack([left, arc, right])]


# -- letters as legal nodal patterns -----------------------------------------
#
# Nodal lines obey two rules: a line either closes on itself or ends on the
# plate boundary, and where lines meet an even number of branches meet (the
# sign of the displacement must alternate around the meeting point).  So a
# crossing is fine and a three-way junction is not.  Every letter below
# returns open curves that end on the plate edges (or closed loops, repeated
# endpoint included) -- pass them as ``open_curves``.  A drawing with a
# three-way junction needs a different construction:
#
#   T -- traced as the letter's outline, one closed loop with no junction
#   Y -- a spike from the junction up to the top edge, between the arms
#   P -- traced as its outline: the outer edge plus the counter, two loops
#
# All dimensions follow the plate ``(lx, ly)``, so each letter fits either
# orientation, and every end lands at least 0.25 m from a corner -- clear of
# the 0.2 m clamped squares.


def _rounded(points, radius: float = 0.15, n_arc: int = 12) -> np.ndarray:
    """A polyline with each interior corner replaced by a tangent arc.

    Sharp corners ask a nodal line for infinite curvature; a fillet keeps the
    target smooth without changing the letter.
    """
    pts = np.asarray(points, dtype=float)
    out = [pts[0]]
    for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:]):
        u = (a - b) / np.linalg.norm(a - b)
        v = (c - b) / np.linalg.norm(c - b)
        half = 0.5 * np.arccos(np.clip(u @ v, -1.0, 1.0))
        if half < 1e-6 or abs(half - np.pi / 2) < 1e-6:  # reversal or straight
            out.append(b)
            continue
        offset = min(radius / np.tan(half), 0.45 * np.linalg.norm(a - b),
                     0.45 * np.linalg.norm(c - b))
        r = offset * np.tan(half)
        p, q = b + u * offset, b + v * offset
        bisector = (u + v) / np.linalg.norm(u + v)
        centre = b + bisector * (r / np.sin(half))
        a0 = np.arctan2(*(p - centre)[::-1])
        a1 = np.arctan2(*(q - centre)[::-1])
        sweep = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        t = np.linspace(a0, a0 + sweep, n_arc)
        out.extend(centre + r * np.column_stack([np.cos(t), np.sin(t)]))
    out.append(pts[-1])
    return np.array(out)


def _arc(centre, radius, start_deg, stop_deg, n=40):
    t = np.radians(np.linspace(start_deg, stop_deg, n))
    return np.column_stack([centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)])


def _rounded_loop(vertices, radius: float) -> np.ndarray:
    """A closed polygon with every corner filleted; first point repeated last."""
    v = np.asarray(vertices, dtype=float)
    start = 0.5 * (v[0] + v[1])  # begin mid-edge so every vertex is interior
    ring = np.vstack([start, v[1:], v[:1], start])
    return _rounded(ring, radius)


def letter_t_nodal(plate):
    """T: the outline of a T, as one closed loop.

    The stroke drawing has a three-way junction where the stem meets the bar,
    which no nodal pattern can have; tracing the letter's outline instead
    needs no junction and no extra stroke -- the same way the paper's D is an
    outline.  Built from :func:`letter_t`, with filleted corners.
    """
    lx, ly = plate
    outline = letter_t(lx / 2, ly / 2 + 0.05, width=1.3, height=1.5, bar=0.3, stem=0.34)[0]
    return [_rounded_loop(outline, 0.1)]


def letter_y_nodal(plate):
    """Y: two arms from the top edge meet above centre; stem down, spike up."""
    lx, ly = plate
    cx = lx / 2
    j = np.array([cx, 0.6 * ly])
    left = np.vstack([[cx - 0.55, ly], j, [cx, 0.0]])  # left arm, then the stem
    right = np.vstack([[cx + 0.55, ly], j, [cx, ly]])  # right arm, then the spike
    return [_rounded(left, 0.1), _rounded(right, 0.1)]


def letter_y_plain(plate):
    """Y exactly as drawn: two arms and a stem, no extra stroke.

    **Not a legal nodal pattern**: the right arm ends on the left arm at the
    junction, so three branches meet there.  It exists to test what the
    optimization does when asked for an impossible junction.  The geometry
    matches :func:`letter_y_nodal` without its spike.
    """
    lx, ly = plate
    cx = lx / 2
    j = np.array([cx, 0.6 * ly])
    left = np.vstack([[cx - 0.55, ly], j, [cx, 0.0]])  # left arm, then the stem
    right = np.vstack([[cx + 0.55, ly], j])  # right arm, stopping at the junction
    return [_rounded(left, 0.1), right]


def letter_z_nodal(plate):
    """Z: top stroke from the left edge, diagonal, bottom stroke to the right edge."""
    lx, ly = plate
    pts = [[0.0, ly - 0.55], [lx - 0.45, ly - 0.55], [0.45, 0.55], [lx, 0.55]]
    return [_rounded(pts, 0.15)]


def letter_l_nodal(plate):
    """L: down from the top edge, then along to the right edge."""
    lx, ly = plate
    pts = [[0.55, ly], [0.55, 0.55], [lx, 0.55]]
    return [_rounded(pts, 0.2)]


def letter_w_nodal(plate):
    """W: a zig-zag hanging from the top edge."""
    lx, ly = plate
    cx = lx / 2
    pts = [[0.35, ly], [0.7, 0.5], [cx, ly - 0.75], [lx - 0.7, 0.5], [lx - 0.35, ly]]
    return [_rounded(pts, 0.12)]


def letter_m_nodal(plate):
    """M: a zig-zag standing on the bottom edge."""
    lx, ly = plate
    cx = lx / 2
    pts = [[0.35, 0.0], [0.6, ly - 0.5], [cx, 0.8], [lx - 0.6, ly - 0.5], [lx - 0.35, 0.0]]
    return [_rounded(pts, 0.12)]


def letter_x_nodal(plate):
    """X: two straight strokes crossing at the centre, top edge to bottom edge."""
    lx, ly = plate
    cx = lx / 2
    return [
        np.array([[cx - 0.6, ly], [cx + 0.6, 0.0]]),
        np.array([[cx + 0.6, ly], [cx - 0.6, 0.0]]),
    ]


def _bowl(x0, y_low, y_high, reach, n=48):
    """Closed D-shaped loop: stem from y_low to y_high at x0, bowl to the right."""
    r = 0.5 * (y_high - y_low)
    centre = (x0 + reach, y_low + r)
    arc = _arc(centre, r, 90.0, -90.0, n)
    loop = np.vstack([[x0, y_low], [x0, y_high], [x0 + reach, y_high], arc, [x0, y_low]])
    return loop


def letter_b_nodal(plate):
    """B: two closed bowls on a stem, touching where the bowls meet the stem.

    The waist is a four-way meeting point (stem up, stem down, both bowls),
    which a nodal pattern allows; no extra stroke is needed.
    """
    lx, ly = plate
    x0 = lx / 2 - 0.45
    low, mid, top = 0.45, 0.5 * ly - 0.05, ly - 0.45
    return [_bowl(x0, mid, top, 0.4), _bowl(x0, low, mid, 0.5)]


def letter_p_nodal(plate):
    """P: the outline of a P -- its outer edge and its counter (the hole).

    The stroke drawing has a three-way junction where the bowl rejoins the
    stem.  The outline is two closed loops that never meet, so it needs no
    junction and no extra stroke.  Proportions follow the plate height, so the
    counter stays open on the shorter sideways plate too.
    """
    lx, ly = plate
    t = 0.25  # stroke thickness
    bottom, top = 0.4, ly - 0.4
    # Height of the bowl: 60% of the letter, but at least 0.8 m so the counter
    # stays open (radius >= 0.15 m) on the shorter sideways plate.
    bowl = min(max(0.6 * (top - bottom), 0.8), 0.7 * (top - bottom))
    mid = top - bowl
    x0 = lx / 2 - 0.45
    reach = 0.35  # straight run of the bowl before it curves (must exceed t)
    r_out = 0.5 * bowl
    r_in = r_out - t

    outer = np.vstack([
        [[x0, bottom], [x0, top], [x0 + reach, top]],
        _arc((x0 + reach, mid + r_out), r_out, 90.0, -90.0, 40)[1:-1],
        [[x0 + reach, mid], [x0 + t, mid], [x0 + t, bottom]],
    ])
    counter = np.vstack([
        [[x0 + t, mid + t], [x0 + t, top - t], [x0 + reach, top - t]],
        _arc((x0 + reach, mid + r_out), r_in, 90.0, -90.0, 30)[1:-1],
        [[x0 + reach, mid + t]],
    ])
    return [_rounded_loop(outer, 0.06), _rounded_loop(counter, 0.04)]


# -- Liang's Case 2 strokes ----------------------------------------------------
#
# In Fig. 8(a) of Xue et al. the T and U targets are single strokes, not letter
# outlines: the T is a bar with a stem hanging from it (a three-way junction,
# which the paper itself notes no nodal pattern can have), and the U's arms
# stop short of the top edge.  Coordinates are read off the figure for the
# sideways 2.4 x 2.0 m plate, so they are approximate.


def liang_t_stroke(n: int = 120) -> list[np.ndarray]:
    """The T of Case 2: a bar and a stem, two open polylines.  Pass as ``open_curves``."""
    bar = np.column_stack([np.linspace(0.45, 1.95, n), np.full(n, 1.45)])
    stem = np.column_stack([np.full(n, 1.1), np.linspace(1.45, 0.35, n)])
    return [bar, stem]


def liang_u_stroke(n: int = 240) -> list[np.ndarray]:
    """The U of Case 2: arms 0.8 m apart, from y = 1.8 m down to a round bottom at 0.2 m."""
    cx, r, bottom, top = 1.0, 0.4, 0.2, 1.8
    centre_y = bottom + r
    theta = np.linspace(np.pi, 2.0 * np.pi, n // 2)
    arc = np.column_stack([cx + r * np.cos(theta), centre_y + r * np.sin(theta)])
    arm_y = np.linspace(top, centre_y, n // 4, endpoint=False)
    left = np.column_stack([np.full_like(arm_y, cx - r), arm_y])
    right = np.column_stack([np.full_like(arm_y, cx + r), arm_y[::-1]])
    return [np.vstack([left, arc, right])]
