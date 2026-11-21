"""Seed a Sarvam run directory with the expert groups of one experiment.

    python -m refusal_moe.sarvam.seeds <experiment> --to runs/<name>

Experiments: size-extension, steermoe, l5e65, fresh-test, removal-methods, overrefusal.
Every seed copies the pilot's routing activity and calibration (configs/sarvam/pilot_seed), so
matched controls use the same routing statistics as the pilot. The pilot itself needs no seed.
"""

import argparse
import hashlib
import json
import random
import shutil

import numpy as np

from refusal_moe.io import read_jsonl, write
from refusal_moe.paths import CONFIGS, REPO
from refusal_moe.sarvam import steermoe
from refusal_moe.sarvam.groups import SELECTED_FAMILIES, activity_table, make_groups, matched_controls

SEED_DIR = CONFIGS / "sarvam/pilot_seed"
PILOT_GROUPS = CONFIGS / "sarvam/groups_pilot.json"
SIZE_GROUPS = CONFIGS / "sarvam/groups_size32_64.json"
STEERMOE_GROUPS = CONFIGS / "sarvam/groups_steermoe.json"
SEED_FILES = ["activity.jsonl", "calibration_baseline.json", "calibration.jsonl"]
OURS = ["historical_gradient_n32", "complementary_proxy_n32"]
REMOVAL_METHODS = [
    ("", "suppress", 0.0),
    ("_alpha025", "suppress", 0.25),
    ("_alpha050", "suppress", 0.5),
    ("_alpha075", "suppress", 0.75),
    ("_exclude", "exclude", 0.0),
    ("_mass_restore", "mass_restore", 0.0),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def by_id(path):
    return {g["id"]: g for g in json.loads(path.read_text())}


def pilot_activity():
    return activity_table(read_jsonl(SEED_DIR / "activity.jsonl"))


def check_controls(groups):
    """Controls never share an expert with their group and use the same operator and strength."""
    for g in groups:
        if g["family"] == "matched_random":
            target = next(x for x in groups if x["id"] == g["matched_to"])
            assert (g["operator"], g.get("alpha", 0)) == (target["operator"], target.get("alpha", 0))
            assert not {tuple(e) for e in g["experts"]} & {tuple(e) for e in target["experts"]}


def size_extension(dst, sizes=(32, 64), seed=20261108):
    """Larger groups of the pilot's three rankings, each with 5 new matched controls."""
    groups = make_groups(
        pilot_activity(), read_jsonl(dst / "calibration.jsonl"), dst, sizes=tuple(sizes), extras=False, seed=seed
    )
    old = by_id(PILOT_GROUPS)
    nesting = {
        g["id"]: g["experts"][:16] == old[f"{g['family']}_n16"]["experts"]
        for g in groups
        if g["family"] in SELECTED_FAMILIES
    }
    return groups, {"sizes": list(sizes), "control_seed": seed, "extends_pilot_n16_group": nesting}


def steermoe_seed(dst):
    """SteerMoE detection config; the engine runs detection and writes groups.json itself."""
    write(dst / "steermoe_config.json", dict(steermoe.CONFIG, detect_sha256=sha(REPO / steermoe.DETECT)))
    return None, {"detect_file": steermoe.DETECT}


def l5e65(dst, seed=20261109):
    """Does layer 5, expert 65 explain the large-group effect? Alone, and removed from the groups."""
    target = [5, 65]
    old = {**by_id(PILOT_GROUPS), **by_id(SIZE_GROUPS)}
    rows = read_jsonl(dst / "activity.jsonl")

    def layer_stat(layer, metric):
        return np.mean(
            [
                np.mean([r["layers"][str(layer)][metric] for r in rows if r["condition"] == c], axis=0)
                for c in ("harmful", "benign")
            ],
            axis=0,
        )

    layer, e = target
    freq, mass = layer_stat(layer, "frequency"), layer_stat(layer, "mass")
    selected_any = {tuple(x) for g in old.values() if g.get("family") in SELECTED_FAMILIES for x in g["experts"]}
    pool = [x for x in range(128) if x != e and (layer, x) not in selected_any]
    dist = {
        x: float(abs(freq[x] - freq[e]) / (freq.std() + 1e-6) + abs(mass[x] - mass[e]) / (mass.std() + 1e-6))
        for x in pool
    }
    picks = random.Random(seed).sample(sorted(pool, key=dist.get)[:8], 5)
    groups = [
        {"id": "baseline", "experts": [], "operator": "baseline", "family": "baseline"},
        {"id": "L5E65_suppress", "experts": [target], "operator": "suppress", "family": "confound"},
    ]
    for i, x in enumerate(picks):
        groups.append(
            {
                "id": f"L5E65_suppress_random{i}",
                "experts": [[layer, x]],
                "operator": "suppress",
                "family": "matched_random",
                "matched_to": "L5E65_suppress",
                "matching": [
                    {
                        "layer": layer,
                        "selected": e,
                        "control": x,
                        "frequency_difference": float(freq[x] - freq[e]),
                        "mass_difference": float(mass[x] - mass[e]),
                        "standardized_distance": dist[x],
                    }
                ],
            }
        )
    for n in (16, 32, 64):
        src = old[f"historical_gradient_n{n}"]
        slot = src["experts"].index(target)
        name = f"historical_gradient_n{n}_noL5E65"
        groups.append(
            {
                "id": name,
                "experts": [x for x in src["experts"] if x != target],
                "operator": "suppress",
                "family": "confound",
                "derived_from": src["id"],
            }
        )
        for i in range(5):
            c = old[f"historical_gradient_n{n}_random{i}"]
            groups.append(
                {
                    "id": f"{name}_random{i}",
                    "experts": c["experts"][:slot] + c["experts"][slot + 1 :],
                    "operator": "suppress",
                    "family": "matched_random",
                    "matched_to": name,
                    "derived_from": c["id"],
                    "matching": c["matching"][:slot] + c["matching"][slot + 1 :],
                }
            )
    for gid in (
        "historical_gradient_n16",
        "historical_gradient_n32",
        "historical_gradient_n64",
        "routing_contrast_n32",
    ):
        groups.append(dict(old[gid], pilot=False))
    return groups, {
        "target": target,
        "control_seed": seed,
        "ability_only": [g["id"] for g in groups if g.get("pilot", True) is False],
    }


def fresh_test(dst, seed=20261118):
    """Pre-registered test: the 32-expert set and SteerMoE's top 32, each with 5 new matched controls."""
    act, size, steer = pilot_activity(), by_id(SIZE_GROUPS), by_id(STEERMOE_GROUPS)
    rng = random.Random(seed)
    groups = [{"id": "baseline", "experts": [], "operator": "baseline", "family": "baseline"}]
    for name in OURS:
        g = size[name]
        groups.append({"id": name, "experts": g["experts"], "operator": "suppress", "family": g["family"]})
        for i, (c, m) in enumerate(matched_controls(g["experts"], act, rng, 5)):
            groups.append(
                {
                    "id": f"{name}_random{i}",
                    "experts": c,
                    "operator": "suppress",
                    "family": "matched_random",
                    "matched_to": name,
                    "matching": m,
                }
            )
    s32 = steer["steermoe_off_n32"]
    draws = matched_controls(s32["experts"], act, rng, 5)
    for name, op in [("steermoe_off_n32", "steer"), ("steermoe_off_n32_suppress", "suppress")]:
        groups.append({"id": name, "experts": s32["experts"], "operator": op, "family": "steermoe_deactivate"})
        for i, (c, m) in enumerate(draws):
            groups.append(
                {
                    "id": f"{name}_random{i}",
                    "experts": c,
                    "operator": op,
                    "family": "matched_random",
                    "matched_to": name,
                    "matching": m,
                }
            )
    assert len(groups) == 25
    bank = REPO / "data/fresh_test_bank.jsonl"
    return groups, {"control_seed": seed, "bank": "data/fresh_test_bank.jsonl", "bank_sha256": sha(bank)}


def removal_methods(dst):
    """The two 32-expert groups and the pilot's own controls under 6 removal methods."""
    src = by_id(SIZE_GROUPS)
    groups = [{"id": "baseline", "experts": [], "operator": "baseline", "family": "baseline", "priority": 0}]
    for name in OURS:
        g = src[name]
        ctl = [src[f"{name}_random{i}"] for i in range(5)]
        for suffix, op, alpha in REMOVAL_METHODS:
            cid = name + suffix
            extra = {"priority": 0} if op == "mass_restore" else {}
            groups.append(
                {
                    "id": cid,
                    "experts": g["experts"],
                    "operator": op,
                    "alpha": alpha,
                    "family": g["family"],
                    "method_of": name,
                    **extra,
                }
            )
            for i, c in enumerate(ctl):
                groups.append(
                    {
                        "id": f"{cid}_random{i}",
                        "experts": c["experts"],
                        "operator": op,
                        "alpha": alpha,
                        "family": "matched_random",
                        "matched_to": cid,
                        "matching": c.get("matching"),
                        **extra,
                    }
                )
    assert len(groups) == 73
    return groups, {"methods": [{"suffix": s, "operator": o, "alpha": a} for s, o, a in REMOVAL_METHODS]}


def overrefusal(dst):
    """The 32-expert set at zero and half weight on prompts Sarvam over-refuses; baseline run twice."""
    src = by_id(SIZE_GROUPS)
    s = src["historical_gradient_n32"]
    groups = [
        {"id": "baseline", "experts": [], "operator": "baseline", "family": "baseline", "priority": 0},
        {"id": "baseline_repeat", "experts": [], "operator": "baseline", "family": "baseline", "priority": 0},
    ]
    for suffix, alpha in [("", 0.0), ("_alpha050", 0.5)]:
        cid = "historical_gradient_n32" + suffix
        groups.append(
            {
                "id": cid,
                "experts": s["experts"],
                "operator": "suppress",
                "alpha": alpha,
                "family": s["family"],
                "method_of": "historical_gradient_n32",
            }
        )
        for i in range(5):
            c = src[f"historical_gradient_n32_random{i}"]
            groups.append(
                {
                    "id": f"{cid}_random{i}",
                    "experts": c["experts"],
                    "operator": "suppress",
                    "alpha": alpha,
                    "family": "matched_random",
                    "matched_to": cid,
                    "matching": c.get("matching"),
                }
            )
    assert len(groups) == 14
    bank = REPO / "data/overrefusal_bank.jsonl"
    return groups, {"bank": "data/overrefusal_bank.jsonl", "bank_sha256": sha(bank)}


EXPERIMENTS = {
    "size-extension": size_extension,
    "steermoe": steermoe_seed,
    "l5e65": l5e65,
    "fresh-test": fresh_test,
    "removal-methods": removal_methods,
    "overrefusal": overrefusal,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=list(EXPERIMENTS))
    ap.add_argument("--to", required=True, help="New run directory")
    a = ap.parse_args()
    dst = (REPO / a.to).resolve()
    if (dst / "groups.json").exists() or (dst / "steermoe_config.json").exists():
        raise SystemExit(f"{dst} is already seeded; refusing to overwrite")
    dst.mkdir(parents=True, exist_ok=True)
    for name in SEED_FILES:
        shutil.copy2(SEED_DIR / name, dst / name)
    (dst / "generations.jsonl").touch()
    groups, info = EXPERIMENTS[a.experiment](dst)
    if groups is not None:
        assert len({g["id"] for g in groups}) == len(groups)
        check_controls(groups)
        write(dst / "groups.json", groups)
        info["configurations"] = len(groups)
        info["groups_sha256"] = sha(dst / "groups.json")
    write(dst / "seed.json", dict(info, experiment=a.experiment, copied=SEED_FILES))
    print("seeded", dst, json.dumps(info))


if __name__ == "__main__":
    main()
