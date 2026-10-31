import numpy as np
import matplotlib.pyplot as plt
from CopCarCTRM import CopCarCTRM
from CopCarEnv import CopCarEnv
from TabularLearning import DynamicQLearning
from TabularLearningCounterFactual import DynamicQLearningCounterFactual
from value_iteration import ValueIteration

if __name__ == "__main__":
    rows = 5
    columns = 5
    probability = 0.9
    environment = CopCarEnv(rows=rows, cols=columns, probability=probability)
    ctrm = CopCarCTRM()
    Gamma = 0.001
    num_episodes = 1000
    episode_length = 100
    alpha = 0.01
    UPDATE_FREQUENCY = 75

    vi = ValueIteration(gamma=Gamma, environment=environment, ctrm=ctrm)
    Value = vi.doVI()

    print(Value)

    all_runs_data_classic = []
    for i in range(10):
        print("Run ", i, " started")
        tabularlearning = DynamicQLearning(alpha=alpha, gamma=Gamma, UPDATE_FREQUENCY=UPDATE_FREQUENCY,
                                           environment=environment, ctrm=ctrm)
        tabular_data = tabularlearning.train(num_episodes=num_episodes, max_episode_length=episode_length)
        all_runs_data_classic.append(tabular_data)
        #print(all_runs_data_classic)
    average_results_classic = [0] * len(all_runs_data_classic[0])
    for i in range(len(all_runs_data_classic[0])):
        sum_val = 0
        for j in range(len(all_runs_data_classic)):
            sum_val += all_runs_data_classic[j][i]
        average_results_classic[i] = sum_val / len(all_runs_data_classic)
        average_results_classic[i] /= Value
    print(average_results_classic)

    plt.plot(np.arange(len(average_results_classic)) * UPDATE_FREQUENCY, average_results_classic)
    plt.xlabel('Episodes')
    plt.ylabel('Agent Value')
    plt.title('Agent Performance Over Time')
    plt.show()

    sum = []
    all_runs_data_counter = []
    for i in range(10):
        print("Run ", i, " started")
        tabularlearningcounter = DynamicQLearningCounterFactual(alpha=alpha, gamma=Gamma,
                                                                UPDATE_FREQUENCY=UPDATE_FREQUENCY,
                                                                environment=environment, ctrm=ctrm)
        tabular_data_counter = tabularlearningcounter.train(num_episodes=num_episodes,
                                                            max_episode_length=episode_length)
        all_runs_data_counter.append(tabular_data_counter)
        #print(all_runs_data_counter)
    average_results_counter = [0] * len(all_runs_data_counter[0])
    for i in range(len(all_runs_data_counter[0])):
        sum_val = 0
        for j in range(len(all_runs_data_counter)):
            sum_val += all_runs_data_counter[j][i]
        average_results_counter[i] = sum_val / len(all_runs_data_counter)
        average_results_counter[i] /= Value
    print(average_results_counter)

    plt.plot(np.arange(len(average_results_counter)) * UPDATE_FREQUENCY, average_results_counter)
    plt.xlabel('Episodes')
    plt.ylabel('Agent Value')
    plt.title('Agent Performance Over Time')
    plt.show()

    plt.plot(np.arange(len(average_results_counter)) * UPDATE_FREQUENCY, average_results_counter,
             label='Counterfactual Data')

    # Plot classic data
    plt.plot(np.arange(len(average_results_classic)) * UPDATE_FREQUENCY, average_results_classic, label='Classic Data')

    # Adding labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Value per step')
    plt.title('Agent Performance Over Time (Tabular Q-learning)')

    # Adding a legend to clarify which line corresponds to which dataset
    plt.legend()

    # Show the plot
    plt.show()