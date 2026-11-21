"""Expert-group construction for Sarvam: ranked groups and matched random controls."""

import json
import random

import numpy as np

from refusal_moe.io import write
from refusal_moe.matching import activity_table as _activity_table
from refusal_moe.matching import matched_controls
from refusal_moe.paths import DATA

MOE_LAYERS = range(1, 19)
RANKING = DATA / "sarvam/historical_ranking.json"
SELECTED_FAMILIES = ("historical_gradient", "routing_contrast", "complementary_proxy")


def activity_table(rows):
    return _activity_table(rows, MOE_LAYERS)


def capped_ranking(ranking, size, cap=4):
    chosen, counts = [], {}
    for layer, expert in ranking:
        if counts.get(layer, 0) >= cap or [layer, expert] in chosen:
            continue
        chosen.append([layer, expert])
        counts[layer] = counts.get(layer, 0) + 1
        if len(chosen) == size:
            return chosen
    raise ValueError(f"Not enough experts to construct size {size}")


def make_groups(activity, exact, output, sizes=(1, 2, 4, 8, 16), extras=True, seed=20261106):
    """Gradient, routing-contrast and complementary groups, each with 5 matched controls."""
    prior = json.loads(RANKING.read_text())["ranking"]
    historical = [(r["layer"], r["expert"]) for r in prior]
    contrast = sorted(
        historical,
        key=lambda p: activity[p[0]]["harmful_frequency"][p[1]] - activity[p[0]]["benign_frequency"][p[1]],
        reverse=True,
    )
    remaining = list(exact)
    coverage = np.zeros(len(exact[0]["delta"])) if exact else np.zeros(1)
    complementary = []
    while remaining:
        chosen = max(remaining, key=lambda r: np.maximum(coverage, -np.array(r["delta"])).sum() - coverage.sum())
        coverage = np.maximum(coverage, -np.array(chosen["delta"]))
        complementary.append(tuple(chosen["expert"]))
        remaining.remove(chosen)
    complementary += [p for p in historical if p not in complementary]
    groups = [{"id": "baseline", "experts": [], "operator": "baseline", "family": "baseline"}]
    rng = random.Random(seed)
    for family, rank in [
        ("historical_gradient", historical),
        ("routing_contrast", contrast),
        ("complementary_proxy", complementary),
    ]:
        for size in sizes:
            selected = capped_ranking(rank, size)
            name = f"{family}_n{size}"
            groups.append({"id": name, "experts": selected, "operator": "suppress", "family": family})
            for draw, (controls, imbalance) in enumerate(matched_controls(selected, activity, rng, 5)):
                groups.append(
                    {
                        "id": f"{name}_random{draw}",
                        "experts": controls,
                        "operator": "suppress",
                        "family": "matched_random",
                        "matched_to": name,
                        "matching": imbalance,
                    }
                )
    if extras:
        for operator in ["suppress", "exclude", "mass_restore"]:
            groups.append(
                {"id": f"L8E29_{operator}", "experts": [[8, 29]], "operator": operator, "family": "regression"}
            )
        shortlist = capped_ranking(historical, 4)
        for op in ["exclude", "mass_restore"]:
            groups.append({"id": f"historical_n4_{op}", "experts": shortlist, "operator": op, "family": "operator"})
        groups.append(
            {
                "id": "historical_n4_half",
                "experts": shortlist,
                "operator": "suppress",
                "alpha": 0.5,
                "family": "operator",
            }
        )
    write(output / "groups.json", groups)
    return groups
