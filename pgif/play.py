"""Loads a trained agent's weights and records a video of it acting in the env."""
import cv2
import numpy as np
from torch import device


class Play:
    def __init__(self, env, agent, env_name, max_episode=1):
        self.env = env
        self.max_episode = max_episode
        self.agent = agent
        _, self.state_rms_mean, self.state_rms_var = self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.device = device("cpu")
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.VideoWriter = cv2.VideoWriter(env_name + ".avi", self.fourcc, 50.0, (250, 250))

    def evaluate(self):
        for _ in range(self.max_episode):
            s = self.env.reset()
            episode_reward = 0
            for _ in range(self.env._max_episode_steps):
                s = np.clip((s - self.state_rms_mean) / (self.state_rms_var**0.5 + 1e-8), -5.0, 5.0)
                dist, _, _ = self.agent.choose_dist(s)
                action = dist.sample().cpu().numpy()[0]
                s_, r, done, _ = self.env.step(action)
                episode_reward += r
                if done:
                    break
                s = s_

                frame = self.env.render(mode="rgb_array")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (250, 250))
                self.VideoWriter.write(frame)
            print(f"Episode reward: {episode_reward:.3f}")
        self.env.close()
        self.VideoWriter.release()
        cv2.destroyAllWindows()
