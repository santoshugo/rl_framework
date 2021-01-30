import pickle
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file
from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.utils.rendertools import RenderTool

from observations import SimpleGraphObservation

f = 'local_tests\Test_30\Level_1.pkl'
with open(f, 'rb') as t:
    x = pickle.load(t)

env = RailEnv(width=1,
              height=1,
              rail_generator=rail_from_file(f),
              schedule_generator=schedule_from_file(f),
              malfunction_generator_and_process_data=malfunction_from_file(f),
              obs_builder_object=SimpleGraphObservation())

obs = env.reset()
print(obs)

env_renderer = RenderTool(env)
env_renderer.render_env(show=True, frames=True, show_observations=False)
input("Press Enter to continue...")