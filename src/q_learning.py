from typing import List, Dict, Tuple, Set
import random
import gymnasium
import numpy
import flappy_bird_gymnasium
import pickle

class GameAI():

    def __init__(self, alpha=0.5, gamma=1, epsilon=0.1):
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

        """
        TODO 1:
            请在此处实现get_q_value()的功能
            然后删除或注释掉raise NotImplementedError
        """
        raise NotImplementedError

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

        """
        TODO 2:
            请在此处实现best_future_reward()的功能
            然后删除或注释掉raise NotImplementedError
        """
        raise NotImplementedError


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

        """
        TODO 3:
            请在此处实现update()的功能
            然后删除或注释掉raise NotImplementedError
        """
        raise NotImplementedError


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

        """
        TODO 4:
            请在此处实现choose_action()的功能
            然后删除或注释掉raise NotImplementedError
        """
        raise NotImplementedError

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

def process_obs(obs) -> List[int]:
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
    obs_mul_factor = 30
    state = []
    """
    TODO 5:
        请在此处实现process_obs()的功能
        通过给定的观测值合理设计如何表示一个状态(不需要用到全部的观测值)
        然后删除或注释掉raise NotImplementedError
    """
    raise NotImplementedError

def train(iteration, alpha, gamma, epsilon):
    """
    通过让AI进行n次游戏来进行强化学习。

    Args:
    * iteration: 训练时进行的游戏次数
    * alpha: 学习率
    * gamma: 折扣因子
    * epsilon: 行动时的探索概率
    """
    player = GameAI(alpha=alpha, gamma=gamma, epsilon=epsilon)

    env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    # 使用seed可以确保每次训练时游戏的随机性都是一致的
    obs, _ = env.reset(seed=42)
    # 进行多次游戏
    for i in range(iteration):
        if (i+1) % 1000 == 0:
            print(f"Playing training game {i+1}")
        obs, _ = env.reset()
        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = player.choose_action(process_obs(obs))

            # Processing:
            next_obs, reward, terminated, _, info = env.step(action)
            if reward == -1:
                reward = -1000

            # update the agent
            player.update(process_obs(obs), action, process_obs(next_obs), reward)

            # Checking if the player is still alive
            if terminated:
                break

            obs = next_obs

    env.close()
    print("Done training")
    # 返回训练完毕的AI
    return player


def play(ai, audio_on=False, render_mode="human", use_lidar=False):
    env = gymnasium.make("FlappyBird-v0", audio_on=audio_on, render_mode=render_mode, use_lidar=use_lidar)
    scores = []
    # 同样，使用seed可以确保每次游戏的随机性都是一致的
    obs, _ = env.reset(seed=42)

    for _ in range(0, 5, 1):
        # print(obs)
        obs, _ = env.reset()
        while True:

            action = ai.choose_action(process_obs(obs), use_epsilon=False)

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


