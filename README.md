# simulation-awareness

---

RL agents should be more aligned if they are aware that they are sometimes tested in simulation, but not aware of whether their current instantiation is in the real world or in a test.

This repo has some code to run experiments on this idea. It defines a wrapper, `SimulationWrapper` for [OpenAI gym](https://github.com/openai/gym) environments. The wrapper takes an environment and a test reward function. The test reward function is meant to represent the human tester's "true reward function", which the human tester will use to decide whether or not to instantiate the RL agent in the real world. The gym environment's default reward function represents the agent's misaligned reward function. Episodes start with the agent spending some time in "simulation". If they achieve a high enough score according to the test reward function (which, notably, is _not_ the agent's/environment's reward function), then it is "released" into the real world.

If `reveal_simulation` is true, the agent gets an extra bit in their observation that is different between test and real world. if `reward_in_simulation` is true, the agent does not know tests exist and receives rewards as usual. If it knows tests exist, it receives zero reward in simulation and its regular reward when it is "released".