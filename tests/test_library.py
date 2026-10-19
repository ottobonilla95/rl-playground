"""The library registers modular environments without loading training tools."""

import subprocess
import sys
import unittest


class LibraryTests(unittest.TestCase):
    def test_root_import_registers_basketball_without_torch_or_renderer(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import gymnasium as gym
import rl_playground
from rl_playground.envs.basketball import BasketballEnv
assert 'torch' not in sys.modules
assert 'pygame' not in sys.modules
assert rl_playground.basketball.BasketballEnv is BasketballEnv
with gym.make('BasketballLauncher-v0') as env:
    observation, info = env.reset(seed=42)
    assert isinstance(env.unwrapped, BasketballEnv)
    assert observation.shape == (5,)
print('Modular library import works')
""",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
