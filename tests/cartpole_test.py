# cartpole_quick.py
import gymnasium as gym
from stable_baselines3 import PPO

# Train headless (no window)
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10_000)

# Show a short run with a window (uses pygame-ce)
test_env = gym.make("CartPole-v1", render_mode="human")
obs, _ = test_env.reset(seed=0)
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = test_env.step(action)
    if term or trunc:
        break
test_env.close()
print("CartPole OK")