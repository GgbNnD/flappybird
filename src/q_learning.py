from typing import List, Set
import random
import gymnasium
import flappy_bird_gymnasium
import pickle

class GameAI():

    def __init__(self, alpha=0.5, gamma=1, epsilon=0.1, rng=None):
        """
        初始化：
        一个字典self.q表示Q-Function，存储从（状态，行动）对到Q-Value的映射，

        Args:
        * alpha: 学习率
        * gamma: 折扣因子
        * epsilon: 行动时的探索概率
        """
        self.q = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng or random

    def save_q(self, path:str):
        """
        根据path保存Q-Function

        Args:
        * path: Q-Function的保存路径
        """
        with open(path, 'wb') as ff:
            pickle.dump(self.q, ff)

    def load_q(self, path:str):
        """
        根据path读取Q-Function

        Args:
        * path: Q-Function文件的读取路径
        """
        with open(path, 'rb') as ff:
            self.q = pickle.load(ff)

    def get_q_value(self, state:List[int], action:int) -> float:
        """
        返回(state, action)对的Q-Value，
        如果self.q中不存在对应的Q-Value，则返回0。

        Args:
        * state: 状态
        * action: 行动

        Returns:
        * (state, action)对的Q-Value
        """

        return self.q.get((tuple(state), action), 0)

    def best_future_reward(self, state:List[int]) -> float:
        """
        给定状态state，考虑该状态中所有可能的（状态，行动）对，返回所有Q-Value的最大值。
        如果（状态，行动）对不在self.q中，则使用0作为Q-Value。
        如果该状态下没有合法行动，则返回0。

        Args:
        * state: 状态

        Returns:
        * 最大的Q-Value
        """

        actions = self.available_actions(state)
        if not actions:
            return 0
        return max(self.get_q_value(state, action) for action in actions)


    def update(self, old_state:List[int], action:int, new_state:List[int], reward):
        """
        给定一个(old_state, action, new_state, reward)样本对，
        使用Q-Learning算法更新Q-Value。

        Args:
        * old_state: 旧状态
        * action: 行动
        * new_state: 新状态
        * reward: 回报
        """

        old_q = self.get_q_value(old_state, action)
        future_reward = self.best_future_reward(new_state)
        new_estimate = reward + self.gamma * future_reward
        self.q[(tuple(old_state), action)] = old_q + self.alpha * (new_estimate - old_q)


    def choose_action(self, state:List[int], use_epsilon=True) -> int:
        """
        给定状态，返回要采取的行动。
        如果epsilon为False，则返回该状态下的最优行动（具有最高Q-Value的行动，如果self.q中不存在则Q-Value为0）。
        如果epsilon为True，则以概率self.epsilon选择一个随机的合法行动，以概率1-self.epsilon选择最优行动。
        如果多个行动具有相同的Q-Value，则可以返回其中任何一个。

        Args:
        * state: 当前的状态
        * use_epsilon: 是否使用epsilon-greedy算法

        Returns:
        * 采取的行动
        """

        actions = sorted(self.available_actions(state))
        if use_epsilon and self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        best_value = max(self.get_q_value(state, action) for action in actions)
        best_actions = [
            action for action in actions
            if self.get_q_value(state, action) == best_value
        ]
        return self.rng.choice(best_actions)

    @classmethod
    def available_actions(cls, state:List[int]) -> Set[int]:
        """
        对给定的状态state，返回该状态下的所有合法行动。
        @classmethod表示这是类方法，因此不需要创建Nim的实例就可以调用该方法。
        调用方式为：GameAI.available_actions(state)。

        Args:
        * state: 当前的状态

        Returns:
        * 所有合法的行动
        """
        # 使用集合set来存储行动，确保集合中的元素不重复
        # 因为Bird的动作在任何状态下都是两个，所以传入的state实际上没有作用
        actions = set()
        actions.add(0)
        actions.add(1) # 1 means flap
        return actions

def process_obs(obs, obs_mul_factor=30) -> List[int]:
    """
    通过obs_mul_factor，将游戏环境返回的各种观测值obs转换成合适的状态值。

    obs[0]: the last pipe's horizontal position
    obs[1]: the last top pipe's vertical position
    obs[2]: the last bottom pipe's vertical position
    obs[3]: the next pipe's horizontal position
    obs[4]: the next top pipe's vertical position
    obs[5]: the next bottom pipe's vertical position
    obs[6]: the next next pipe's horizontal position
    obs[7]: the next next top pipe's vertical position
    obs[8]: the next next bottom pipe's vertical position
    obs[9]: player's vertical position
    obs[10]: player's vertical velocity
    obs[11]: player's rotation

    Args:
    * obs: Flappy Bird游戏环境返回的各种观测值

    Return:
    * 根据状态设计，将当前的观测值转换成对应的状态state
    """
    player_x = 0.2
    pipe_width = 52 / 288

    # 如果第一个管道已经完全位于小鸟身后，就转而使用下一个管道。
    pipe_start = 3 if obs[0] + pipe_width < player_x else 0
    pipe_x = obs[pipe_start]
    pipe_top = obs[pipe_start + 1]
    pipe_bottom = obs[pipe_start + 2]
    player_y = obs[9]
    player_v = obs[10]

    gap_center = (pipe_top + pipe_bottom) / 2
    state = [
        pipe_x - player_x,
        player_y - gap_center,
        pipe_bottom - player_y,
        player_v,
    ]
    return [int(value * obs_mul_factor) for value in state]

def train(
    iteration,
    alpha,
    gamma,
    epsilon,
    obs_mul_factor=30,
    seed=42,
    death_penalty=-1000,
    progress_interval=1000,
    max_steps_per_episode=None,
):
    """
    通过让AI进行n次游戏来进行强化学习。

    Args:
    * iteration: 训练时进行的游戏次数
    * alpha: 学习率
    * gamma: 折扣因子
    * epsilon: 行动时的探索概率
    """
    rng = random.Random(seed)
    player = GameAI(alpha=alpha, gamma=gamma, epsilon=epsilon, rng=rng)

    env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    # 使用seed可以确保每次训练时游戏的随机性都是一致的
    obs, _ = env.reset(seed=seed)
    # 进行多次游戏
    for i in range(iteration):
        if progress_interval and (i+1) % progress_interval == 0:
            print(f"Playing training game {i+1}")
        obs, _ = env.reset()
        steps = 0
        while True:
            # Next action:
            # (feed the observation to your agent here)
            state = process_obs(obs, obs_mul_factor)
            action = player.choose_action(state)

            # Processing:
            next_obs, reward, terminated, _, info = env.step(action)
            steps += 1
            if reward == -1:
                reward = death_penalty

            # update the agent
            player.update(state, action, process_obs(next_obs, obs_mul_factor), reward)

            # Checking if the player is still alive
            if terminated:
                break
            if max_steps_per_episode and steps >= max_steps_per_episode:
                break

            obs = next_obs

    env.close()
    print("Done training")
    # 返回训练完毕的AI
    return player


def play(
    ai,
    audio_on=False,
    render_mode="human",
    use_lidar=False,
    episodes=5,
    obs_mul_factor=30,
    seed=42,
):
    env = gymnasium.make("FlappyBird-v0", audio_on=audio_on, render_mode=render_mode, use_lidar=use_lidar)
    scores = []
    # 同样，使用seed可以确保每次游戏的随机性都是一致的
    obs, _ = env.reset(seed=seed)

    for _ in range(episodes):
        # print(obs)
        obs, _ = env.reset()
        while True:

            action = ai.choose_action(process_obs(obs, obs_mul_factor), use_epsilon=False)

            # Processing:
            obs, _, done , _, info = env.step(action)
            """
            这里将Obs的输出注释掉了，如有调试需要，可以自行开启
            """
            # print(f"Obs: {obs}\n" f"Score: {info['score']}\n")

            if done:
                scores.append(info['score'])
                print(f"This try gets {info['score']} score(s).")
                break

    env.close()
    print(f'The average score(s) of Q-Function: {sum(scores) / len(scores)}')
    assert obs.shape == env.observation_space.shape
    return scores
