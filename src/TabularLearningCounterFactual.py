import numpy as np
import math
from collections import deque


class DynamicQLearningCounterFactual:
    def __init__(self, alpha=0.1, gamma=0.001, epsilon=0.1, UPDATE_FREQUENCY=50, environment=None, ctrm=None,
                 decay_rate=0.05, reward_shaping=False):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.max_epsilon = epsilon
        self.min_epsilon = 0.1
        self.decay_rate = decay_rate
        self.epsilon = self.max_epsilon
        self.UPDATE_FREQUENCY = UPDATE_FREQUENCY
        self.evaluation_results = [0]
        self.epsilon_decay = decay_rate
        self.env = environment
        self.ctrm = ctrm
        self.reward_shaping = reward_shaping
        # Information for checking the value of the strategy
        self.V = {}
        self.ctrmV = {}
        self.states = self.fill_states()
        self.ctrm_states = tuple(self.ctrm.states)
        self.fill_vtable()
        self.fillctrmV()

    def fillctrmV(self):
        if self.reward_shaping:
            for ctrm_state in self.ctrm_states:
                self.ctrmV[ctrm_state] = 0  # for CTRM value iteration

    def fill_states(self):
        initial_state = self.env.initstate
        state_set = {initial_state}  # This set stores all the states seen so far
        # Initialize the queue with the initial state
        queue = deque([initial_state])  # Initializes the queue with the initial state
        while queue:  # Runs until the queue is empty
            current_state = queue.popleft()  # pops the state from the queue
            for next_state in self.env.next_states(current_state):  # Gets the next states of the current state
                if next_state not in state_set:  # Checks if the state is seen or not
                    state_set.add(next_state)  # Adds the unseen state to the state set
                    queue.append(next_state)  # Adds to the queue to check for unseen next states
        return state_set

    def fill_vtable(self):
        for state in self.states:
            for ctrm_state in self.ctrm.states:
                s = state + (ctrm_state,)
                self.V[s] = 0

    def startegy_analaysis(self):
        enable = True
        initstate = self.env.initstate + (self.ctrm.initstate,)  # Initial state of the product
        while enable:
            delta = 0
            for state in self.V:  # Each state is encoded as (environment state, ctrm state)
                v = self.V[state]  # Previous value
                a = self.pick_best_action(state, self.env.actions)
                ctrm_state = state[-1]  # Get the CTRM state
                env_state = state[:-1]  # Get the environment state
                rate = self.ctrm.get_rate_counterfactual(ctrm_state, env_state, a)  # Gets the rate
                next_states = self.env.next_state(env_state, a)  # Gets the next state of the environment
                for next_state, probability in next_states.items():
                    action_value = 0
                    reward, ctrm_next = self.ctrm.transition_function_counterfactual(ctrm_state, next_state)
                    next_state1 = next_state + (ctrm_next,)
                    if reward is not None and rate is not None:
                        time = 1 / rate
                        # action_value += (probability * (reward + math.exp(-1 * time * self.gamma) * self.V[next_state1]))
                        action_value += (probability * (reward + rate / (self.gamma + rate) * self.V[next_state1]))

                self.V[state] = action_value
                delta = max(delta, abs(v - self.V[state]))
            if delta < 0.01:
                enable = False
        return self.V[initstate]

    def get_q_value(self, state, action, available_actions):
        if state not in self.q_table:
            self.q_table[state] = {}
        for act in available_actions:
            if act not in self.q_table[state]:
                self.q_table[state][act] = 0
        # if action not in self.q_table[state]:
        #     self.q_table[state][action] = 0  # Initialize unseen state-action pairs to 0
        return self.q_table[state][action]

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)  # Random action
        else:
            current_actions = self.q_table.get(state, {})
            if not current_actions:  # If no actions for this state, choose randomly
                return np.random.choice(available_actions)
            return max(current_actions, key=current_actions.get)  # Best action based on current Q-values

    def pick_best_action(self, state, available_actions):
        # print("In pick action")
        current_actions = self.q_table.get(state, {})
        if not current_actions:  # If no actions for this state, choose randomly
            # print("State not seen")
            return available_actions[0]  # The first action is chosen always
        return max(current_actions, key=current_actions.get)

    def update_q_table(self, state, action, reward, sampled_time, next_state, available_actions):
        # Get the best Q value for the next state
        next_best_action_q_value = max(
            self.get_q_value(next_state, a, available_actions=available_actions) for a in available_actions)

        # Current Q value
        current_q_value = self.get_q_value(state, action, available_actions=available_actions)

        # Q-learning formula
        rate = 1 / sampled_time
        new_q_value = current_q_value + self.alpha * (
                    reward + rate / (rate + self.gamma) * next_best_action_q_value - current_q_value)
        self.q_table[state][action] = new_q_value

    def train(self, num_episodes, max_episode_length):
        sum_perfomance = 0
        for episode in range(num_episodes):
            if episode % 1000 == 0:
                 print("Episode:", episode)

            env_state = self.env.reset()
            ctrm_state = self.ctrm.reset()
            for i in range(max_episode_length):
                available_actions = self.env.actions
                action = self.choose_action(env_state + (ctrm_state,), available_actions)

                rate = self.ctrm.get_rate(env_state, action)
                if rate is None:
                    break
                env_state1, sampled_time = self.env.step(action, rate)
                reward = self.ctrm.transitionfunction(
                    self.env.state)  # transition in the ctrm which gives the new state and the reward
                if reward is None:
                    break
                ctrm_state1 = self.ctrm.state  # new ctrm state
                previous_state = env_state + (ctrm_state,)
                next_state = env_state1 + (ctrm_state1,)
                # self.update_q_table(previous_state, action, reward, sampled_time, next_state, self.env.actions)
                self.add_counterfactual_experience(self.ctrm, env_state, action, env_state1, self.env.actions)
                # if reward > 0:
                #     break

                env_state = self.env.state
                ctrm_state = self.ctrm.state
            if (episode + 1) % self.UPDATE_FREQUENCY == 0:
                sum_perfomance = self.get_average(sum_perfomance, (episode + 1) / self.UPDATE_FREQUENCY)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
        return self.evaluation_results

    def trainwithconvergence(self, max_episode_length, value, threshold, max_episodes=100000):
        sum_perfomance = 0
        termination = 0
        episode = 0
        if self.reward_shaping:
            self.ctrm_vi()
        while termination == 0 and episode <= max_episodes:

            # if episode % 1000 == 0:
            #     print("Episode:", episode)
            env_state = self.env.reset()
            ctrm_state = self.ctrm.reset()

            for i in range(max_episode_length):
                available_actions = self.env.actions
                action = self.choose_action(env_state + (ctrm_state,), available_actions)
                rate = self.ctrm.get_rate(env_state, action)
                if rate is None:
                    break
                env_state1, sampled_time = self.env.step(action, rate)
                reward = self.ctrm.transitionfunction(
                    self.env.state)  # transition in the ctrm which gives the new state and the reward
                if reward is None:
                    break
                ctrm_state1 = self.ctrm.state  # new ctrm state
                previous_state = env_state + (ctrm_state,)
                next_state = env_state1 + (ctrm_state1,)
                # self.update_q_table(previous_state, action, reward, sampled_time, next_state, self.env.actions)
                self.add_counterfactual_experience(self.ctrm, env_state, action, env_state1, self.env.actions)
                # if reward > 0:
                #     break

                env_state = self.env.state
                ctrm_state = self.ctrm.state

            if (episode + 1) % self.UPDATE_FREQUENCY == 0:
                sum_perfomance = self.get_average(sum_perfomance, (episode + 1) / self.UPDATE_FREQUENCY, value)
                # print(f"episode: {episode}, values given {self.evaluation_results[-1] / value}")
                if self.evaluation_results[-1] > threshold:
                    termination = 1
                    break
            # print(f"Completed episode {episode}")
            episode += 1
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
        return self.evaluation_results

    def get_average(self, sum_perfomance, step, vi_value=None):
        agent_value = self.get_agent_value()
        self.fill_vtable
        sum_perfomance += agent_value
        value = sum_perfomance / step
        if vi_value is not None and vi_value != 0:
            value = value / vi_value
        # print(f"Episode {step}: Agent Value = {agent_value}")
        self.evaluation_results.append(value)
        return sum_perfomance

    def get_agent_value(self):
        self.env.reset()
        self.ctrm.reset()
        return self.startegy_analaysis()

    def add_counterfactual_experience(self, ctrm, state, action, next_state, available_actions):
        for ctrmstate in ctrm.states:
            rate = ctrm.get_rate_counterfactual(ctrmstate, state, action)
            if rate is not None:
                reward, ctrm_nextstate = ctrm.transition_function_counterfactual(ctrmstate, next_state)
                time = 1 / rate
                if reward is not None and ctrm_nextstate is not None:
                    if self.reward_shaping:
                        reward += self.getrewardshaping(ctrmstate, ctrm_nextstate, time)
                    previous_state = state + (ctrmstate,)
                    next_state1 = next_state + (ctrm_nextstate,)
                    self.update_q_table(previous_state, action, reward, time, next_state1, available_actions)

    def ctrm_vi(self):
        enable = True
        initstate = self.ctrm.initstate  # Initial state of the
        while enable:
            delta = 0
            for state in self.ctrmV:  # Iterate through each state of the CTRM
                v = self.ctrmV[state]  # Previous value
                next_states = self.ctrm.next_states(state)  # Get the next states
                if next_states is not None:
                    action_value = 0
                    for next_state in next_states:  # Get the value of taking each next state
                        reward = self.ctrm.transition_VI(state, next_state)
                        if reward is not None:
                            # print(f"Reward is {reward}")
                            value = reward + 0.7 * self.ctrmV[next_state]
                            action_value = max(value, action_value)  # Take the action value
                    self.ctrmV[state] = action_value
                    delta = max(delta, abs(v - self.ctrmV[state]))
            if delta < 0.01:
                enable = False
                # for state in self.ctrmV:
                #     self.ctrmV[state] = -1 * self.ctrmV[state]
                # print(f"Value of state {state} = {self.ctrmV[state]}")

    def getrewardshaping(self, ctrm_current, ctrm_next, time):  # gets the reward shaping reward
        if self.ctrmV[ctrm_current] is not None and self.ctrmV[ctrm_next] is not None:
            # rate = round(1/time,5)
            # reward =  ((rate/(rate + self.gamma)) * self.ctrmV[ctrm_next]) - self.ctrmV[ctrm_current]
            rate = 1 / time
            reward = rate / (rate + self.gamma) * self.ctrmV[ctrm_next] - self.ctrmV[ctrm_current]
        else:
            reward = 0
        # print(f"Reward from {ctrm_current} to {ctrm_next} is {reward}.")
        return reward





