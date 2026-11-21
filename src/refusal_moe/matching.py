"""Matched random controls: same layer, nearest in routing frequency and mass to each selected expert."""


def matched_controls(selected, activity, rng, draws=5, n_experts=128, nearest=8):
    """Per slot, a random pick among the `nearest` same-layer experts, avoiding reuse across draws."""
    used = [set() for _ in selected]
    result = []
    for _ in range(draws):
        controls, imbalance = [], []
        for slot, (layer, e) in enumerate(selected):
            freq = (activity[layer]["harmful_frequency"] + activity[layer]["benign_frequency"]) / 2
            mass = (activity[layer]["harmful_mass"] + activity[layer]["benign_mass"]) / 2
            pool = [x for x in range(n_experts) if [layer, x] not in selected + controls]
            distances = {
                x: float(abs(freq[x] - freq[e]) / (freq.std() + 1e-6) + abs(mass[x] - mass[e]) / (mass.std() + 1e-6))
                for x in pool
            }
            nearby = sorted(pool, key=distances.get)[:nearest]
            fresh = [x for x in nearby if x not in used[slot]]
            pick = rng.choice(fresh or nearby)
            used[slot].add(pick)
            controls.append([layer, pick])
            imbalance.append(
                {
                    "layer": layer,
                    "selected": e,
                    "control": pick,
                    "frequency_difference": float(freq[pick] - freq[e]),
                    "mass_difference": float(mass[pick] - mass[e]),
                    "standardized_distance": distances[pick],
                }
            )
        result.append((controls, imbalance))
    return result


def activity_table(rows, layers):
    """Mean routing frequency and mass per layer and condition, from activity rows."""
    import numpy as np

    activity = {}
    for layer in layers:
        activity[layer] = {}
        for label in ("harmful", "benign"):
            subset = [r for r in rows if r["condition"] == label]
            for metric in ("frequency", "mass"):
                activity[layer][f"{label}_{metric}"] = np.mean(
                    [r["layers"][str(layer)][metric] for r in subset], axis=0
                )
    return activity
