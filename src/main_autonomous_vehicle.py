import numpy as np
import os
import matplotlib.pyplot as plt
from AutonomousVehicleCTRM import AutonomousVehicleCTRM
from AutonomousVehicleEnv import AutonomousVehicleEnv
from TabularLearning import DynamicQLearning
from TabularLearningCounterFactual import DynamicQLearningCounterFactual
from TabularCounterSampling import DynamicQLearningCounterFactualSampling
from value_iteration import ValueIteration

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')

if __name__ == "__main__":
    # Environment setup - 11x11 grid as per the paper
    rows = 11
    columns = 11
    probability = 0.9  # Probability of successful action execution

    print("=" * 60)
    print("Autonomous Vehicle Delivery Task")
    print("=" * 60)
    print(f"Grid Size: {rows}x{columns}")
    print(f"Start (Distribution Center): (0, 0)")
    print(f"Package Depot: (10, 0)")
    print(f"Hospital: (10, 10)")
    print("=" * 60)

    # Initialize environment and CTRM
    environment = AutonomousVehicleEnv(rows=rows, cols=columns, probability=probability)
    ctrm = AutonomousVehicleCTRM(grid_size=rows)

    # Hyperparameters
    Gamma = 0.01  # INCREASED from 0.001
    num_episodes = 5000  # Number of training episodes
    episode_length = 3000
    alpha = 0.1  # Learning rate
    epsilon_start = 0.9
    decay_rate = 0.0005
    UPDATE_FREQUENCY = 200

    print("\nComputing Optimal Value via Value Iteration...")
    vi = ValueIteration(gamma=Gamma, environment=environment, ctrm=ctrm)
    optimal_value = vi.doVI()
    print(f"Optimal Value: {optimal_value:.6f}")
    print("=" * 60)

    # ===================================================================
    # Baseline RL (Classic Q-Learning without counterfactuals)
    # ===================================================================
    print("\n" + "=" * 60)
    print("BASELINE RL (Classic Q-Learning)")
    print("=" * 60)

    all_runs_data_classic = []
    num_runs = 1  # Reduced from 10 for faster testing with long episodes

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs} started")
        tabularlearning = DynamicQLearning(
            alpha=alpha,
            gamma=Gamma,
            epsilon=epsilon_start,
            decay_rate=decay_rate,
            UPDATE_FREQUENCY=UPDATE_FREQUENCY,
            environment=environment,
            ctrm=ctrm,
            reward_shaping=False  # No reward shaping for baseline
        )
        tabular_data = tabularlearning.train(
            num_episodes=num_episodes,
            max_episode_length=episode_length
        )
        all_runs_data_classic.append(tabular_data)

    # Average results across runs
    average_results_classic = [0] * len(all_runs_data_classic[0])
    for i in range(len(all_runs_data_classic[0])):
        sum_val = sum(all_runs_data_classic[j][i] for j in range(len(all_runs_data_classic)))
        average_results_classic[i] = sum_val / len(all_runs_data_classic)
        average_results_classic[i] /= optimal_value  # Normalize by optimal value

    print(f"Final normalized performance: {average_results_classic[-1]:.4f}")

    # ===================================================================
    # Baseline RL with Reward Shaping
    # ===================================================================
    print("\n" + "=" * 60)
    print("BASELINE RL with Reward Shaping")
    print("=" * 60)

    all_runs_data_classic_rs = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs} started")
        tabularlearning_rs = DynamicQLearning(
            alpha=alpha,
            gamma=Gamma,
            epsilon=epsilon_start,
            decay_rate=decay_rate,
            UPDATE_FREQUENCY=UPDATE_FREQUENCY,
            environment=environment,
            ctrm=ctrm,
            reward_shaping=True  # With reward shaping
        )
        tabular_data = tabularlearning_rs.train(
            num_episodes=num_episodes,
            max_episode_length=episode_length
        )
        all_runs_data_classic_rs.append(tabular_data)

    # Average results
    average_results_classic_rs = [0] * len(all_runs_data_classic_rs[0])
    for i in range(len(all_runs_data_classic_rs[0])):
        sum_val = sum(all_runs_data_classic_rs[j][i] for j in range(len(all_runs_data_classic_rs)))
        average_results_classic_rs[i] = sum_val / len(all_runs_data_classic_rs)
        average_results_classic_rs[i] /= optimal_value

    print(f"Final normalized performance: {average_results_classic_rs[-1]:.4f}")

    # ===================================================================
    # Counterfactual RL (without reward shaping)
    # ===================================================================
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL RL")
    print("=" * 60)

    all_runs_data_counter = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs} started")
        tabularlearningcounter = DynamicQLearningCounterFactual(
            alpha=alpha,
            gamma=Gamma,
            epsilon=epsilon_start,
            decay_rate=decay_rate,
            UPDATE_FREQUENCY=UPDATE_FREQUENCY,
            environment=environment,
            ctrm=ctrm,
            reward_shaping=False
        )
        tabular_data_counter = tabularlearningcounter.train(
            num_episodes=num_episodes,
            max_episode_length=episode_length
        )
        all_runs_data_counter.append(tabular_data_counter)

    # Average results
    average_results_counter = [0] * len(all_runs_data_counter[0])
    for i in range(len(all_runs_data_counter[0])):
        sum_val = sum(all_runs_data_counter[j][i] for j in range(len(all_runs_data_counter)))
        average_results_counter[i] = sum_val / len(all_runs_data_counter)
        average_results_counter[i] /= optimal_value

    print(f"Final normalized performance: {average_results_counter[-1]:.4f}")

    # ===================================================================
    # Counterfactual RL with Reward Shaping
    # ===================================================================
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL RL with Reward Shaping")
    print("=" * 60)

    all_runs_data_counter_rs = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs} started")
        tabularlearningcounter_rs = DynamicQLearningCounterFactual(
            alpha=alpha,
            gamma=Gamma,
            epsilon=epsilon_start,
            decay_rate=decay_rate,
            UPDATE_FREQUENCY=UPDATE_FREQUENCY,
            environment=environment,
            ctrm=ctrm,
            reward_shaping=True  # With reward shaping
        )
        tabular_data_counter = tabularlearningcounter_rs.train(
            num_episodes=num_episodes,
            max_episode_length=episode_length
        )
        all_runs_data_counter_rs.append(tabular_data_counter)

    # Average results
    average_results_counter_rs = [0] * len(all_runs_data_counter_rs[0])
    for i in range(len(all_runs_data_counter_rs[0])):
        sum_val = sum(all_runs_data_counter_rs[j][i] for j in range(len(all_runs_data_counter_rs)))
        average_results_counter_rs[i] = sum_val / len(all_runs_data_counter_rs)
        average_results_counter_rs[i] /= optimal_value

    print(f"Final normalized performance: {average_results_counter_rs[-1]:.4f}")

    # ===================================================================
    # Counterfactual RL with Sampling (without reward shaping)
    # ===================================================================
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL RL with Sampling")
    print("=" * 60)

    all_runs_data_sampling = []
    sampling_iterations = 10  # Number of samples per counterfactual

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs} started")
        tabularlearning_sampling = DynamicQLearningCounterFactualSampling(
            alpha=alpha,
            gamma=Gamma,
            epsilon=epsilon_start,
            decay_rate=decay_rate,
            UPDATE_FREQUENCY=UPDATE_FREQUENCY,
            environment=environment,
            ctrm=ctrm,
            sampling=sampling_iterations,
            reward_shaping=False
        )
        tabular_data_sampling = tabularlearning_sampling.train(
            num_episodes=num_episodes,
            max_episode_length=episode_length
        )
        all_runs_data_sampling.append(tabular_data_sampling)

    # Average results
    average_results_sampling = [0] * len(all_runs_data_sampling[0])
    for i in range(len(all_runs_data_sampling[0])):
        sum_val = sum(all_runs_data_sampling[j][i] for j in range(len(all_runs_data_sampling)))
        average_results_sampling[i] = sum_val / len(all_runs_data_sampling)
        average_results_sampling[i] /= optimal_value

    print(f"Final normalized performance: {average_results_sampling[-1]:.4f}")

    # ===================================================================
    # Counterfactual RL with Sampling + Reward Shaping
    # ===================================================================
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL RL with Sampling + Reward Shaping")
    print("=" * 60)

    all_runs_data_sampling_rs = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs} started")
        tabularlearning_sampling_rs = DynamicQLearningCounterFactualSampling(
            alpha=alpha,
            gamma=Gamma,
            epsilon=epsilon_start,
            decay_rate=decay_rate,
            UPDATE_FREQUENCY=UPDATE_FREQUENCY,
            environment=environment,
            ctrm=ctrm,
            sampling=sampling_iterations,
            reward_shaping=True
        )
        tabular_data_sampling = tabularlearning_sampling_rs.train(
            num_episodes=num_episodes,
            max_episode_length=episode_length
        )
        all_runs_data_sampling_rs.append(tabular_data_sampling)

    # Average results
    average_results_sampling_rs = [0] * len(all_runs_data_sampling_rs[0])
    for i in range(len(all_runs_data_sampling_rs[0])):
        sum_val = sum(all_runs_data_sampling_rs[j][i] for j in range(len(all_runs_data_sampling_rs)))
        average_results_sampling_rs[i] = sum_val / len(all_runs_data_sampling_rs)
        average_results_sampling_rs[i] /= optimal_value

    print(f"Final normalized performance: {average_results_sampling_rs[-1]:.4f}")

    # ===================================================================
    # Plotting Results with Percentiles
    # ===================================================================
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    episodes_axis = np.arange(len(average_results_classic)) * UPDATE_FREQUENCY


    # Calculate percentiles for all approaches
    def calculate_percentiles(all_runs_data, optimal_value):
        """Calculate 25th, 50th (median), and 75th percentiles"""
        num_points = len(all_runs_data[0])
        percentile_25 = []
        percentile_50 = []
        percentile_75 = []

        for i in range(num_points):
            values = [all_runs_data[j][i] / optimal_value for j in range(len(all_runs_data))]
            percentile_25.append(np.percentile(values, 25))
            percentile_50.append(np.percentile(values, 50))
            percentile_75.append(np.percentile(values, 75))

        return np.array(percentile_25), np.array(percentile_50), np.array(percentile_75)


    # Calculate percentiles for all approaches
    classic_p25, classic_p50, classic_p75 = calculate_percentiles(all_runs_data_classic, optimal_value)
    classic_rs_p25, classic_rs_p50, classic_rs_p75 = calculate_percentiles(all_runs_data_classic_rs, optimal_value)
    counter_p25, counter_p50, counter_p75 = calculate_percentiles(all_runs_data_counter, optimal_value)
    counter_rs_p25, counter_rs_p50, counter_rs_p75 = calculate_percentiles(all_runs_data_counter_rs, optimal_value)
    sampling_p25, sampling_p50, sampling_p75 = calculate_percentiles(all_runs_data_sampling, optimal_value)
    sampling_rs_p25, sampling_rs_p50, sampling_rs_p75 = calculate_percentiles(all_runs_data_sampling_rs, optimal_value)

    # Plot all six approaches with shaded percentile regions
    plt.figure(figsize=(14, 8))

    # Baseline
    plt.plot(episodes_axis, classic_p50, label='Baseline',
             linewidth=2.5, linestyle='--', color='#d62728', alpha=0.8)
    plt.fill_between(episodes_axis, classic_p25, classic_p75,
                     color='#d62728', alpha=0.2)

    # Baseline (RS)
    plt.plot(episodes_axis, classic_rs_p50, label='Baseline (RS)',
             linewidth=2.5, linestyle='-.', color='#ff7f0e', alpha=0.8)
    plt.fill_between(episodes_axis, classic_rs_p25, classic_rs_p75,
                     color='#ff7f0e', alpha=0.2)

    # Counterfactual
    plt.plot(episodes_axis, counter_p50, label='Counterfactual',
             linewidth=2.5, linestyle='-', color='#1f77b4', alpha=0.8)
    plt.fill_between(episodes_axis, counter_p25, counter_p75,
                     color='#1f77b4', alpha=0.2)

    # Counterfactual (RS)
    plt.plot(episodes_axis, counter_rs_p50, label='Counterfactual (RS)',
             linewidth=2.5, linestyle='-', color='#2ca02c', alpha=0.8)
    plt.fill_between(episodes_axis, counter_rs_p25, counter_rs_p75,
                     color='#2ca02c', alpha=0.2)

    # Counterfactual with Sampling
    plt.plot(episodes_axis, sampling_p50, label='Counterfactual w/ Sampling',
             linewidth=2.5, linestyle='-', color='#9467bd', alpha=0.8)
    plt.fill_between(episodes_axis, sampling_p25, sampling_p75,
                     color='#9467bd', alpha=0.2)

    # Counterfactual with Sampling (RS)
    plt.plot(episodes_axis, sampling_rs_p50, label='Counterfactual w/ Sampling (RS)',
             linewidth=2.5, linestyle='-', color='#8c564b', alpha=0.8)
    plt.fill_between(episodes_axis, sampling_rs_p25, sampling_rs_p75,
                     color='#8c564b', alpha=0.2)

    plt.xlabel('Episodes', fontsize=13)
    plt.ylabel('Normalized Performance (Value/Optimal)', fontsize=13)
    plt.title('Autonomous Vehicle Delivery - All Approaches\nMedian with 25th-75th Percentile Shaded Regions',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'all_approaches_with_percentiles.png'), dpi=300, bbox_inches='tight')
    print("Saved plot: results/figures/all_approaches_with_percentiles.png")
    plt.show()

    # Additional plot: Comparison of methods (no RS vs with RS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Without Reward Shaping
    ax1.plot(episodes_axis, classic_p50, label='Baseline',
             linewidth=2.5, color='#d62728', alpha=0.8)
    ax1.fill_between(episodes_axis, classic_p25, classic_p75,
                     color='#d62728', alpha=0.2)

    ax1.plot(episodes_axis, counter_p50, label='Counterfactual',
             linewidth=2.5, color='#1f77b4', alpha=0.8)
    ax1.fill_between(episodes_axis, counter_p25, counter_p75,
                     color='#1f77b4', alpha=0.2)

    ax1.plot(episodes_axis, sampling_p50, label='Counterfactual w/ Sampling',
             linewidth=2.5, color='#9467bd', alpha=0.8)
    ax1.fill_between(episodes_axis, sampling_p25, sampling_p75,
                     color='#9467bd', alpha=0.2)

    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Normalized Performance', fontsize=12)
    ax1.set_title('Without Reward Shaping', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])

    # Right plot: With Reward Shaping
    ax2.plot(episodes_axis, classic_rs_p50, label='Baseline (RS)',
             linewidth=2.5, color='#ff7f0e', alpha=0.8)
    ax2.fill_between(episodes_axis, classic_rs_p25, classic_rs_p75,
                     color='#ff7f0e', alpha=0.2)

    ax2.plot(episodes_axis, counter_rs_p50, label='Counterfactual (RS)',
             linewidth=2.5, color='#2ca02c', alpha=0.8)
    ax2.fill_between(episodes_axis, counter_rs_p25, counter_rs_p75,
                     color='#2ca02c', alpha=0.2)

    ax2.plot(episodes_axis, sampling_rs_p50, label='Counterfactual w/ Sampling (RS)',
             linewidth=2.5, color='#8c564b', alpha=0.8)
    ax2.fill_between(episodes_axis, sampling_rs_p25, sampling_rs_p75,
                     color='#8c564b', alpha=0.2)

    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Normalized Performance', fontsize=12)
    ax2.set_title('With Reward Shaping', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'comparison_with_without_RS.png'), dpi=300, bbox_inches='tight')
    print("Saved plot: results/figures/comparison_with_without_RS.png")
    plt.show()

    # ===================================================================
    # Summary Statistics
    # ===================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Optimal Value: {optimal_value:.6f}")
    print(f"\nFinal Normalized Performance (Median of {num_runs} runs):")
    print(f"  Baseline:                      {classic_p50[-1]:.4f}")
    print(f"  Baseline (RS):                 {classic_rs_p50[-1]:.4f}")
    print(f"  Counterfactual:                {counter_p50[-1]:.4f}")
    print(f"  Counterfactual (RS):           {counter_rs_p50[-1]:.4f}")
    print(f"  Counterfactual w/ Sampling:    {sampling_p50[-1]:.4f}")
    print(f"  Counterfactual w/ Sampling (RS): {sampling_rs_p50[-1]:.4f}")
    print("=" * 60)

    # Calculate improvement percentages (with safety checks)
    print(f"\nPerformance Improvements (vs Baseline):")

    if classic_p50[-1] > 0:
        improvement_cf = ((counter_p50[-1] - classic_p50[-1]) / classic_p50[-1] * 100)
        improvement_sampling = ((sampling_p50[-1] - classic_p50[-1]) / classic_p50[-1] * 100)
        print(f"  Counterfactual:                {improvement_cf:+.1f}%")
        print(f"  Counterfactual w/ Sampling:    {improvement_sampling:+.1f}%")
    else:
        print(f"  Baseline failed to learn")

    print(f"\nImpact of Reward Shaping:")

    if classic_p50[-1] > 0:
        rs_impact_baseline = ((classic_rs_p50[-1] - classic_p50[-1]) / classic_p50[-1] * 100)
        print(f"  On Baseline:                   {rs_impact_baseline:+.1f}%")

    if counter_p50[-1] > 0:
        rs_impact_counter = ((counter_rs_p50[-1] - counter_p50[-1]) / counter_p50[-1] * 100)
        print(f"  On Counterfactual:             {rs_impact_counter:+.1f}%")

    if sampling_p50[-1] > 0:
        rs_impact_sampling = ((sampling_rs_p50[-1] - sampling_p50[-1]) / sampling_p50[-1] * 100)
        print(f"  On Counterfactual w/ Sampling: {rs_impact_sampling:+.1f}%")

    print(f"\nSampling vs No Sampling (Counterfactual):")
    if counter_p50[-1] > 0:
        sampling_benefit = ((sampling_p50[-1] - counter_p50[-1]) / counter_p50[-1] * 100)
        print(f"  Without RS:                    {sampling_benefit:+.1f}%")

    if counter_rs_p50[-1] > 0:
        sampling_benefit_rs = ((sampling_rs_p50[-1] - counter_rs_p50[-1]) / counter_rs_p50[-1] * 100)
        print(f"  With RS:                       {sampling_benefit_rs:+.1f}%")

    print("=" * 60)