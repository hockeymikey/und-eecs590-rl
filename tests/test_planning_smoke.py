from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl590.envs.windy_chasm import WindyChasmMDP
from rl590.dp.planning import policy_iteration, q_value_policy_iteration, value_iteration
from rl590.model.belief import TabularModelBelief, collect_transitions


class PlanningSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = WindyChasmMDP(p_center=0.1)

    def test_value_iteration_shapes(self) -> None:
        V, Q, policy, _, _ = value_iteration(self.env.P, self.env.R, self.env.gamma, max_iterations=200)
        self.assertEqual(V.shape, (self.env.num_states,))
        self.assertEqual(Q.shape, (self.env.num_states, self.env.num_actions))
        self.assertEqual(policy.shape, (self.env.num_states,))

    def test_policy_iteration_shapes(self) -> None:
        V, Q, policy, _, _ = policy_iteration(self.env.P, self.env.R, self.env.gamma, max_iterations=50)
        self.assertEqual(V.shape, (self.env.num_states,))
        self.assertEqual(Q.shape, (self.env.num_states, self.env.num_actions))
        self.assertEqual(policy.shape, (self.env.num_states,))

    def test_q_value_iteration_shapes(self) -> None:
        V, Q, policy, _, _ = q_value_policy_iteration(self.env.P, self.env.R, self.env.gamma, max_iterations=200)
        self.assertEqual(V.shape, (self.env.num_states,))
        self.assertEqual(Q.shape, (self.env.num_states, self.env.num_actions))
        self.assertEqual(policy.shape, (self.env.num_states,))

    def test_belief_update_and_estimate_shapes(self) -> None:
        belief = TabularModelBelief(self.env.num_states, self.env.num_actions)
        transitions = collect_transitions(
            self.env,
            episodes=5,
            max_steps_per_episode=20,
            epsilon=1.0,
            seed=0,
            policy=None,
        )
        belief.update_batch(transitions)

        P_hat, R_hat = belief.estimated_mdp()
        self.assertEqual(P_hat.shape, (self.env.num_actions, self.env.num_states, self.env.num_states))
        self.assertEqual(R_hat.shape, (self.env.num_states, self.env.num_actions))
        self.assertGreater(belief.num_updates, 0)
        self.assertTrue((P_hat >= 0.0).all())
        self.assertTrue((P_hat <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
