import sys
import os

# Go up two levels: experiments -> stl-go -> MARLlib root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from marllib import marl

RESULTS_DIR = "../results"

# 1. Create SMAC environment
# Change map_name to the SMAC map you want, e.g. "8m", "3s5z", "MMM2"
env = marl.make_env(environment_name="smac", map_name="8m")

# 2. Pick algorithm + hyperparameters
mappo = marl.algos.mappo(hyperparam_source="smac")

# 3. Build model
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# 4. Train (with evaluation)
mappo.fit(
    env,
    model,
    stop={"timesteps_total": 100000},
    checkpoint_freq=50,
    share_policy="group",
    local_dir=RESULTS_DIR,
)

# 4. Train (skip evaluation)
# mappo.fit(
#     env,
#     model,
#     stop={"timesteps_total": 100000},
#     checkpoint_freq=50,
#     share_policy="group",
#     local_dir=RESULTS_DIR,
#     evaluation_interval=None
# )