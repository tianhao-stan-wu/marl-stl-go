import sys
import os

# Go up two levels: experiments -> stl-go -> MARLlib root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from marllib import marl

RESULTS_DIR = "../results"

# 1. Create environment
env = marl.make_env(environment_name="lbf", map_name="default_map")

# 2. Pick algorithm + hyperparameters
mappo = marl.algos.mappo(hyperparam_source="common")

# 3. Build model
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# 4. Train (with evaluation)
mappo.fit(
    env, model,
    stop={"timesteps_total": 100000},
    checkpoint_freq=50,
    share_policy="group",
    local_dir=RESULTS_DIR,
)

# 4. Train (skip evaluation)
# mappo.fit(
#     env, model,
#     stop={"timesteps_total": 100000},
#     checkpoint_freq=50,
#     share_policy="group",
#     local_dir=RESULTS_DIR,
#     evaluation_interval=None   # <-- skips evaluation, avoids the crash
# )
