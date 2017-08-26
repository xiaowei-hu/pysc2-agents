from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


def run_loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  try:
    while True:
      num_frames = 0
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        timesteps = env.step(actions)
        # Only for a single player!
        is_done = (num_frames >= max_frames) or timesteps[0].last()
        yield [last_timesteps[0], actions[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)
