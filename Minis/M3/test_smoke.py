#!/usr/bin/env python3
"""Smoke tests for the reactor environment and agents."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from reactor_env import ReactorConfig, ReactorEnv
from td_agent import TDConfig, TDAgent
from fa_agent import FAConfig, FAAgent


class TestReactorEnv(unittest.TestCase):

    def setUp(self):
        self.env = ReactorEnv(ReactorConfig())
        self.rng = default_rng(0)

    def test_reset_returns_valid_bin(self):
        obs = self.env.reset(self.rng)
        self.assertIsInstance(obs, (int, np.integer))
        self.assertGreaterEqual(obs, 0)
        self.assertLess(obs, self.env.num_states)

    def test_step_returns_correct_tuple(self):
        obs = self.env.reset(self.rng)
        next_obs, reward, done = self.env.simulate_step(obs, 0, self.rng)
        self.assertIsInstance(next_obs, (int, np.integer))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, (bool, np.bool_))

    def test_num_actions(self):
        cfg = ReactorConfig(k=2)
        env = ReactorEnv(cfg)
        self.assertEqual(env.num_actions, 5)  # {-2,-1,0,1,2}

    def test_num_states(self):
        cfg = ReactorConfig(n_bins=30)
        env = ReactorEnv(cfg)
        self.assertEqual(env.num_states, 30)

    def test_meltdown_termination(self):
        """Force reactivity up until meltdown."""
        cfg = ReactorConfig(
            mu_min=9.0, mu_max=10.0, init_spread=0.5,
            sigma_process=0.0, sigma_obs=0.01,
            k=2, alpha=0.0, delta=0.5,  # strong drift, no rod effect
        )
        env = ReactorEnv(cfg)
        rng = default_rng(42)
        env.reset(rng)
        done = False
        for _ in range(100):
            _, _, done = env.simulate_step(0, 2, rng)  # action index 2 = 0 rods
            if done:
                break
        self.assertTrue(done, "Expected meltdown termination")

    def test_horizon_termination(self):
        """With safe parameters, episode should last full horizon."""
        cfg = ReactorConfig(
            horizon=20,
            mu_min=0, mu_max=100,  # very wide range, no meltdown
            sigma_process=0.01, sigma_obs=0.01,
            delta=0.0,
        )
        env = ReactorEnv(cfg)
        rng = default_rng(0)
        obs = env.reset(rng)
        steps = 0
        for _ in range(50):
            obs, _, done = env.simulate_step(obs, 2, rng)  # action=0
            steps += 1
            if done:
                break
        self.assertEqual(steps, 20, "Expected horizon termination at 20 steps")

    def test_bin_centers_shape(self):
        centers = self.env.bin_centers()
        self.assertEqual(len(centers), self.env.num_states)

    def test_action_labels(self):
        labels = self.env.action_labels()
        self.assertEqual(len(labels), self.env.num_actions)


class TestTDAgent(unittest.TestCase):

    def test_q_table_shape(self):
        env = ReactorEnv(ReactorConfig())
        agent = TDAgent(env, TDConfig(episodes=10))
        agent.train()
        self.assertEqual(agent.Q.shape, (env.num_states, env.num_actions))

    def test_sarsa_tracking(self):
        env = ReactorEnv(ReactorConfig())
        agent = TDAgent(env, TDConfig(algorithm="sarsa_lambda", episodes=10))
        agent.train()
        self.assertEqual(len(agent.episode_returns), 10)
        self.assertEqual(len(agent.episode_meltdowns), 10)

    def test_qlearning_tracking(self):
        env = ReactorEnv(ReactorConfig())
        agent = TDAgent(env, TDConfig(algorithm="qlearning", episodes=10))
        agent.train()
        self.assertEqual(len(agent.episode_returns), 10)

    def test_evaluate(self):
        env = ReactorEnv(ReactorConfig())
        agent = TDAgent(env, TDConfig(episodes=10))
        agent.train()
        result = agent.evaluate(episodes=5)
        self.assertIn("mean_return", result)
        self.assertIn("meltdown_rate", result)

    def test_save_load_roundtrip(self):
        env = ReactorEnv(ReactorConfig())
        agent = TDAgent(env, TDConfig(episodes=10))
        agent.train()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_agent.npz"
            agent.save(path)

            agent2 = TDAgent(env)
            agent2.load(path)
            np.testing.assert_array_equal(agent.Q, agent2.Q)
            self.assertEqual(len(agent.episode_returns), len(agent2.episode_returns))


class TestFAAgent(unittest.TestCase):

    def test_feature_shape(self):
        env = ReactorEnv(ReactorConfig())
        agent = FAAgent(env, FAConfig(n_rbf_centers=15))
        phi = agent._features(5, 2)
        self.assertEqual(len(phi), 15 * env.num_actions)

    def test_feature_sparsity(self):
        """Only the action-block should be nonzero."""
        env = ReactorEnv(ReactorConfig())
        cfg = FAConfig(n_rbf_centers=10)
        agent = FAAgent(env, cfg)
        phi = agent._features(5, 1)
        # Block for action 1 should be nonzero
        block = phi[10:20]
        self.assertTrue(np.any(block != 0))
        # Other blocks should be zero
        self.assertTrue(np.all(phi[:10] == 0))
        self.assertTrue(np.all(phi[20:] == 0))

    def test_train_and_q_table(self):
        env = ReactorEnv(ReactorConfig())
        agent = FAAgent(env, FAConfig(episodes=10))
        agent.train()
        Q = agent.q_table()
        self.assertEqual(Q.shape, (env.num_states, env.num_actions))

    def test_save_load_roundtrip(self):
        env = ReactorEnv(ReactorConfig())
        agent = FAAgent(env, FAConfig(episodes=10))
        agent.train()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_fa.npz"
            agent.save(path)

            agent2 = FAAgent(env, FAConfig(episodes=10))
            agent2.load(path)
            np.testing.assert_array_almost_equal(agent.w, agent2.w)


if __name__ == "__main__":
    unittest.main()
