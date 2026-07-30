"""Training loop: rollout collection, GAE, and the PPO-style actor/critic update."""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions import normal
from torch.utils.tensorboard import SummaryWriter

from pgif.networks import MLP, BackwardsLSTM
from pgif.utils import RunningMeanStd


class Train:
    def __init__(self, env, test_env, env_name, n_iterations, agent, epochs, mini_batch_size,
                 epsilon, horizon, initialAlpha, initialBeta):
        self.env = env
        self.env_name = env_name
        self.test_env = test_env
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))
        self.alpha = initialAlpha
        self.beta = initialBeta
        self.increase_inits = 0.000001

        self.rewards_history = []
        self.actor_losses = []
        self.running_reward = 0
        self.start_time = time.time()

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs, values, log_probs, meanZs, stdZs, backwards):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            indices = indices.astype(int)
            yield (states[indices], actions[indices], returns[indices], advs[indices], values[indices],
                   log_probs[indices], meanZs[indices], stdZs[indices], [backwards[i] for i in indices])

    def train(self, states, actions, advs, values, log_probs, meanZs, stdZs, backwards):
        values = np.vstack(values[:-1])
        log_probs = np.vstack(log_probs)
        returns = advs + values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        actions = np.vstack(actions)

        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        mlp = MLP(n_states, n_actions)
        lf = nn.MSELoss()
        op = torch.optim.Adam(mlp.parameters(), lr=0.001)

        meanZs = np.array(meanZs)
        stdZs = np.array(stdZs)

        actor_loss = torch.zeros(1)
        critic_loss = torch.zeros(1)

        for epoch in range(self.epochs):
            for state, action, return_, adv, old_value, old_log_prob, meanZ, stdZ, backward in self.choose_mini_batch(
                self.mini_batch_size, states, actions, returns, advs, values, log_probs, meanZs, stdZs, backwards
            ):
                actor_loss = 0
                critic_loss = 0
                lossAxCalc = 0
                for index in range(len(state)):
                    s = torch.Tensor(state[index]).to(self.agent.device)
                    a = torch.Tensor(action[index]).to(self.agent.device)
                    r = torch.Tensor(return_[index]).to(self.agent.device)
                    ad = torch.Tensor(adv[index]).to(self.agent.device)
                    ov = torch.Tensor(old_value[index]).to(self.agent.device)
                    olp = torch.Tensor(old_log_prob[index]).to(self.agent.device)
                    mz = torch.Tensor(meanZ[index]).to(self.agent.device)
                    sz = torch.Tensor(stdZ[index]).to(self.agent.device)
                    bck = backward[index]

                    v = self.agent.critic(s)
                    critic_loss += self.agent.critic_loss(v, r)

                    meanZPG = bck[0]
                    stdZPG = bck[1]
                    distPG = normal.Normal(meanZPG, stdZPG)
                    actionPG = distPG.sample().cpu().numpy()[0]
                    new_log_prob = distPG.log_prob(torch.Tensor(actionPG)).detach()
                    meanAx, stdAx = mlp.forward(s)
                    lossAxCalc = lossAxCalc + lf(meanAx, meanZPG)
                    lossKL = self.kl_divergence_gaussian(mz, sz, meanZPG, stdZPG)

                    ratio = (new_log_prob - olp).exp()
                    actor_loss = actor_loss + self.compute_actor_loss(ratio, ad, lossAxCalc, lossKL)

                    self.actor_losses.append(actor_loss.item())

                a_loss = actor_loss.clone()
                a_loss.requires_grad_()

                self.agent.optimize(a_loss, critic_loss)

        return actor_loss, critic_loss

    @staticmethod
    def kl_divergence_gaussian(mu1, std1, mu2, std2):
        kl = 0.5 * (torch.pow(std1 / std2, 2) + torch.pow((mu2 - mu1) / std2, 2) - 1 + 2 * torch.log(std2 / std1))
        return kl.mean()

    def step(self):
        state = self.env.reset()
        step_rewards = []
        actor_losses = []

        for iteration in range(1, self.n_iterations + 1):
            meanZs = []
            stdZs = []
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []

            for t in range(self.horizon):
                state = np.clip((state - self.state_rms.mean) / (self.state_rms.var**0.5 + 1e-8), -5, 5)
                dist, mean, std = self.agent.choose_dist(state)
                action = dist.sample().cpu().numpy()[0]
                log_prob = dist.log_prob(torch.Tensor(action))
                value = self.agent.get_value(state)
                next_state, reward, done, _ = self.env.step(action)

                meanZs.append(mean)
                stdZs.append(std)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = next_state

            next_state = np.clip((next_state - self.state_rms.mean) / (self.state_rms.var**0.5 + 1e-8), -5, 5)
            next_value = self.agent.get_value(next_state) * (1 - done)
            values.append(next_value)

            advs = self.get_gae(rewards, values, dones)
            states = np.vstack(states)

            backwards = self.backwardsPass(states, actions)

            actor_loss, critic_loss = self.train(states, actions, advs, values, log_probs, meanZs, stdZs, backwards)
            self.alpha += self.increase_inits
            self.beta += self.increase_inits
            self.agent.schedule_lr()
            eval_rewards = evaluate_model(self.agent, self.test_env, self.state_rms, self.agent.action_bounds)
            self.state_rms.update(states)
            self.print_logs(iteration, actor_loss, critic_loss, eval_rewards)

            step_rewards.append(eval_rewards)
            actor_losses.append(actor_loss.item())

        self._plot(step_rewards, actor_losses)

    def _plot(self, step_rewards, actor_losses):
        df = pd.DataFrame({
            "Step": range(1, self.n_iterations + 1),
            "Reward": step_rewards,
            "Actor Loss": actor_losses,
        })

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        df_mean = df.groupby("Step").mean()
        df_std = df.groupby("Step").std()
        plt.plot(df_mean.index, df_mean["Reward"], label="Mean Reward")
        plt.fill_between(df_mean.index, df_mean["Reward"] - df_std["Reward"], df_mean["Reward"] + df_std["Reward"], alpha=0.3)
        plt.xlabel("Step")
        plt.ylabel("Mean Episodic Reward")
        plt.title(self.env_name)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(df["Step"], df["Actor Loss"], label="Actor Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Actor Loss")
        plt.title(self.env_name)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def backwardsPass(self, states, actions):
        backwards = []
        backwardsStates = torch.Tensor(states[::-1].copy()).to(self.agent.device)
        backwardsActions = torch.Tensor(np.array(actions)[::-1].copy()).to(self.agent.device)

        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]

        lstm = BackwardsLSTM(n_states, n_actions)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

        for index, state in enumerate(backwardsStates):
            mean, std = lstm(state)
            backwards.append([mean.detach(), std.detach()])

            loss = loss_function(mean, backwardsActions[index].reshape(mean.shape))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return backwards[::-1]

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        advs = []
        gae = 0
        dones = dones + [0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        advs.reverse()
        return np.vstack(advs)

    def compute_actor_loss(self, ratio, adv, lossAx, lossKL):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()
        loss -= self.alpha * lossAx
        loss -= self.beta * lossKL
        return loss

    def print_logs(self, iteration, actor_loss, critic_loss, eval_rewards):
        if iteration == 1:
            self.running_reward = eval_rewards
        else:
            self.running_reward = self.running_reward * 0.99 + eval_rewards * 0.01

        if iteration % 25 == 0:
            print(f"Iter: {iteration} | "
                  f"Ep_Reward: {eval_rewards:.3f} | "
                  f"Running_reward: {self.running_reward:.3f} | "
                  f"Actor_Loss: {actor_loss:.3f} | "
                  f"Critic_Loss: {critic_loss:.3f} | "
                  f"Iter_duration: {time.time() - self.start_time:.3f} | "
                  f"lr: {self.agent.actor_scheduler.get_last_lr()}")
            self.agent.save_weights(iteration, self.state_rms)

        with SummaryWriter(self.env_name + "/logs") as writer:
            writer.add_scalar("Episode running reward", self.running_reward, iteration)
            writer.add_scalar("Episode reward", eval_rewards, iteration)
            writer.add_scalar("Actor loss", actor_loss, iteration)
            writer.add_scalar("Critic loss", critic_loss, iteration)


def evaluate_model(agent, env, state_rms, action_bounds):
    total_rewards = 0
    s = env.reset()
    done = False
    while not done:
        s = np.clip((s - state_rms.mean) / (state_rms.var**0.5 + 1e-8), -5.0, 5.0)
        dist, _, _ = agent.choose_dist(s)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        if total_rewards < 0:
            total_rewards = 0
        s = next_state
    return total_rewards
