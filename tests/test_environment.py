"""Eight environment tests (SPEC.md §5). Run: python -m tests.test_environment

Each test is standalone; the __main__ runner reports t1..t8 PASS/FAIL so a
single broken invariant is immediately identifiable. t6 (determinism) and
t8 (info-key contract) guard the two failure modes that silently corrupt
experiments, so they must never be skipped.
"""
from __future__ import annotations

import copy
import traceback

import numpy as np

from src.agents.baseline_agents import BASELINE_REGISTRY, make_baseline
from src.environment.demand_generator import DemandGenerator
from src.environment.grid_env import GridEnv, load_config
from src.environment.supply_model import SupplyModel

CFG = load_config()


def t1_gymnasium_contract():
    env = GridEnv(copy.deepcopy(CFG))
    obs, info = env.reset(seed=0)
    assert obs.shape == (5, 14) and obs.dtype == np.float32
    assert isinstance(info, dict)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (5,)
    for t in range(168):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(r, float)
        assert terminated is False
        assert truncated == (t == 167), f"truncation wrong at t={t}"


def t2_demand_model():
    gen = DemandGenerator(copy.deepcopy(CFG))
    rng = np.random.default_rng(0)
    for h in range(168):
        d = gen.sample(h, month=6, rng=rng)
        assert d.shape == (5,) and np.all(d > 0)
    # Diurnal peaks: highest mean demand within ±1h of 07:00 and 19:00.
    hours = np.arange(24)
    total = np.array([gen.mean_demand(h, month=6).sum() for h in hours])
    evening_argmax = int(np.argmax(total))
    assert abs(evening_argmax - 19) <= 1, f"evening peak at {evening_argmax}"
    morning = total[:12]
    morning_argmax = int(np.argmax(morning))
    assert abs(morning_argmax - 7) <= 1, f"morning peak at {morning_argmax}"


def t3_supply_model():
    model = SupplyModel(copy.deepcopy(CFG))
    rng = np.random.default_rng(1)
    model.reset(month=6)
    for _ in range(1000):
        s = model.sample(rng)
        assert s >= 0
    # Shock recovery: force a shock, verify exponential decay with no new shocks.
    model.shock = 0.3
    model.shock_prob = 0.0
    model.sample(rng)
    assert np.isclose(model.shock, 0.3 * (1 - model.recovery_rate))
    model.sample(rng)
    assert np.isclose(model.shock, 0.3 * (1 - model.recovery_rate) ** 2)


def t4_reward_components():
    # Deficit regime: supply mean far below total demand (~2280 MW base).
    cfg = copy.deepcopy(CFG)
    cfg["supply"]["mean_mw"] = 1200
    cfg["supply"]["std_mw"] = 1.0
    cfg["supply"]["shock_prob_per_hour"] = 0.0
    env = GridEnv(cfg)
    env.reset(seed=3)
    # NoShedding under deficit => large imbalance penalty.
    _, _, _, _, info = env.step([0] * 5)
    rc = info["reward_components"]
    assert all(v <= 0 for v in rc.values()), rc
    assert rc["r_imb"] < -0.5, f"expected large imbalance penalty, got {rc['r_imb']}"
    assert info["weighted_unserved_energy"] == 0.0
    # Full shedding => zero imbalance but large WUE penalty.
    _, _, _, _, info = env.step([4] * 5)
    rc = info["reward_components"]
    assert all(v <= 0 for v in rc.values()), rc
    assert rc["r_imb"] == 0.0
    assert rc["r_eff"] < -0.5, f"expected large WUE penalty, got {rc['r_eff']}"
    assert info["weighted_unserved_energy"] > 1000  # MWh, ~criticality-weighted total demand


def t5_fairness_term():
    env = GridEnv(copy.deepcopy(CFG))
    env.reset(seed=4)
    # Identical shed levels every step => equal outage rates => std 0.
    for _ in range(30):
        _, _, _, _, info = env.step([2] * 5)
    assert info["reward_components"]["r_fair"] == 0.0
    assert np.std(info["zone_outage_rates"]) == 0.0
    # Unequal shedding => strictly negative fairness term.
    for _ in range(10):
        _, _, _, _, info = env.step([4, 0, 0, 0, 0])
    assert info["reward_components"]["r_fair"] < 0.0


def t6_determinism():
    def rollout(seed):
        env = GridEnv(copy.deepcopy(CFG))
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(99)  # fixed action sequence
        all_obs, rewards = [obs], []
        for _ in range(168):
            obs, r, _, _, _ = env.step(rng.integers(0, 5, size=5))
            all_obs.append(obs)
            rewards.append(r)
        return np.stack(all_obs), np.array(rewards)

    o1, r1 = rollout(42)
    o2, r2 = rollout(42)
    assert np.array_equal(o1, o2), "observations differ for same seed"
    assert np.array_equal(r1, r2), "rewards differ for same seed"
    o3, _ = rollout(43)
    assert not np.array_equal(o1, o3), "different seeds gave identical rollouts"


def t7_baseline_sanity():
    for name in BASELINE_REGISTRY:
        env = GridEnv(copy.deepcopy(CFG))
        policy = make_baseline(name, copy.deepcopy(CFG))
        obs, _ = env.reset(seed=7)
        policy.reset()
        done = False
        while not done:
            actions = policy.act(obs)
            assert env.action_space.contains(np.asarray(actions))
            obs, _, _, done, _ = env.step(actions)
    # Priority ordering: Accra (crit 1.00, idx 0) never shed more than
    # Northern (crit 0.55, idx 4) — deficit regime to force shedding.
    cfg = copy.deepcopy(CFG)
    cfg["supply"]["mean_mw"] = 1500
    env = GridEnv(cfg)
    policy = make_baseline("priority", cfg)
    obs, _ = env.reset(seed=8)
    policy.reset()
    for _ in range(168):
        actions = np.asarray(policy.act(obs))
        assert actions[0] <= actions[4], f"priority shed Accra before Northern: {actions}"
        obs, _, _, _, _ = env.step(actions)


def t8_interface_contract():
    env = GridEnv(copy.deepcopy(CFG))
    env.reset(seed=8)
    _, _, _, _, info = env.step([1, 2, 3, 0, 4])
    assert "weighted_unserved_energy" in info, "MISSING t8 KEY weighted_unserved_energy"
    assert "zone_shed_fraction" in info, "MISSING t8 KEY zone_shed_fraction"
    wue = info["weighted_unserved_energy"]
    assert isinstance(wue, float) and wue > 0
    zsf = np.asarray(info["zone_shed_fraction"])
    assert zsf.shape == (5,)
    assert np.allclose(zsf, [0.25, 0.5, 0.75, 0.0, 1.0])


TESTS = [
    ("t1 gymnasium contract", t1_gymnasium_contract),
    ("t2 demand model", t2_demand_model),
    ("t3 supply model", t3_supply_model),
    ("t4 reward components", t4_reward_components),
    ("t5 fairness term", t5_fairness_term),
    ("t6 determinism", t6_determinism),
    ("t7 baseline sanity", t7_baseline_sanity),
    ("t8 interface contract", t8_interface_contract),
]


def main() -> int:
    passed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(TESTS)} tests passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
