"""Culvert pricing shared by edges, hairpins and fill crossings (brief section 7)."""

import numba


@numba.njit(cache=True, inline="always")
def culvert_cost(area, breaks, costs):
    for i in range(breaks.shape[0]):
        if area < breaks[i]:
            return costs[i]
    return costs[costs.shape[0] - 1]


@numba.njit(cache=True, inline="always")
def _run_step(is_ch, a, in_run, run_max, run_ts, first, breaks, costs):
    """Advance the channel-run state by one point of an edge's cell sequence.

    A run of consecutive channel points gets one culvert (for the largest
    contributing area in the run). A run that touches an end node is shared
    with the neighboring edge there and charged half by each, so a culvert
    is paid once and the charge is the same in both directions.
    Returns (in_run, run_max, run_ts, charge).
    """
    charge = 0.0
    if is_ch:
        if in_run:
            run_max = max(run_max, a)
        else:
            in_run = True
            run_max = a
            run_ts = first
    elif in_run:
        charge = culvert_cost(run_max, breaks, costs) * (0.5 if run_ts else 1.0)
        in_run = False
    return in_run, run_max, run_ts, charge
