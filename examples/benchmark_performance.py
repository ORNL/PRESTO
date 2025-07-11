"""
PRESTO Performance Benchmarks

This module provides comprehensive benchmarking tools for PRESTO algorithms
across different data types, sizes, and privacy requirements.
"""

import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from ornl_presto import (
    get_noise_generators,
    calculate_utility_privacy_score,
    evaluate_algorithm_confidence,
    recommend_top3,
)


class PRESTOBenchmark:
    """Comprehensive benchmarking suite for PRESTO algorithms."""

    def __init__(self):
        self.results = []
        self.noise_generators = get_noise_generators()

    def generate_test_data(
        self, data_type: str, size: int, seed: int = 42
    ) -> torch.Tensor:
        """Generate different types of test data."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        if data_type == "normal":
            return torch.randn(size)
        elif data_type == "uniform":
            return torch.rand(size) * 100
        elif data_type == "exponential":
            return torch.tensor(np.random.exponential(2.0, size), dtype=torch.float32)
        elif data_type == "bimodal":
            data1 = torch.randn(size // 2) * 0.5 + 2
            data2 = torch.randn(size - size // 2) * 0.5 - 2
            return torch.cat([data1, data2])
        elif data_type == "sparse":
            data = torch.zeros(size)
            indices = torch.randint(0, size, (size // 10,))
            data[indices] = torch.randn(len(indices)) * 10
            return data
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def benchmark_algorithm_performance(
        self,
        algorithms: List[str] = None,
        data_sizes: List[int] = None,
        data_types: List[str] = None,
        epsilon_values: List[float] = None,
    ) -> pd.DataFrame:
        """Benchmark algorithm performance across different conditions."""

        if algorithms is None:
            algorithms = list(self.noise_generators.keys())
        if data_sizes is None:
            data_sizes = [100, 500, 1000, 5000]
        if data_types is None:
            data_types = ["normal", "uniform", "exponential", "bimodal"]
        if epsilon_values is None:
            epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]

        results = []
        total_tests = (
            len(algorithms) * len(data_sizes) * len(data_types) * len(epsilon_values)
        )
        test_count = 0

        print(f"Running {total_tests} benchmark tests...")

        for algorithm in algorithms:
            for size in data_sizes:
                for data_type in data_types:
                    for epsilon in epsilon_values:
                        test_count += 1
                        if test_count % 20 == 0:
                            print(
                                f"Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.1f}%)"
                            )

                        try:
                            # Generate test data
                            data = self.generate_test_data(data_type, size)

                            # Time the algorithm
                            start_time = time.time()
                            noise_fn = self.noise_generators[algorithm]
                            private_data = noise_fn(data, epsilon)
                            execution_time = time.time() - start_time

                            # Calculate utility score
                            utility_score = calculate_utility_privacy_score(
                                data, algorithm, epsilon
                            )

                            # Calculate basic statistics
                            original_mean = data.mean().item()
                            private_mean = torch.mean(private_data).item()
                            mean_error = abs(original_mean - private_mean)

                            original_std = data.std().item()
                            private_std = torch.std(private_data).item()
                            std_error = abs(original_std - private_std)

                            # Memory usage estimation
                            memory_mb = (
                                (data.numel() + private_data.numel()) * 4 / (1024**2)
                            )  # float32 = 4 bytes

                            results.append(
                                {
                                    "algorithm": algorithm,
                                    "data_size": size,
                                    "data_type": data_type,
                                    "epsilon": epsilon,
                                    "execution_time": execution_time,
                                    "utility_score": utility_score,
                                    "mean_error": mean_error,
                                    "std_error": std_error,
                                    "memory_mb": memory_mb,
                                    "mean_relative_error": mean_error
                                    / abs(original_mean)
                                    if original_mean != 0
                                    else float("inf"),
                                    "std_relative_error": std_error / original_std
                                    if original_std != 0
                                    else float("inf"),
                                }
                            )

                        except Exception as e:
                            print(
                                f"Error with {algorithm}, {data_type}, size={size}, ε={epsilon}: {e}"
                            )
                            results.append(
                                {
                                    "algorithm": algorithm,
                                    "data_size": size,
                                    "data_type": data_type,
                                    "epsilon": epsilon,
                                    "execution_time": float("inf"),
                                    "utility_score": float("-inf"),
                                    "mean_error": float("inf"),
                                    "std_error": float("inf"),
                                    "memory_mb": float("inf"),
                                    "mean_relative_error": float("inf"),
                                    "std_relative_error": float("inf"),
                                }
                            )

        return pd.DataFrame(results)

    def benchmark_scalability(self, sizes: List[int] = None) -> pd.DataFrame:
        """Benchmark algorithm scalability with increasing data sizes."""

        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000, 10000]

        algorithms = ["DP_Gaussian", "DP_Laplace", "DP_Exponential"]  # Core algorithms
        epsilon = 1.0
        results = []

        print("Running scalability benchmarks...")

        for size in sizes:
            print(f"Testing size: {size}")
            data = self.generate_test_data("normal", size)

            for algorithm in algorithms:
                try:
                    # Multiple runs for statistical significance
                    times = []
                    for run in range(5):
                        start_time = time.time()
                        noise_fn = self.noise_generators[algorithm]
                        private_data = noise_fn(data, epsilon)
                        times.append(time.time() - start_time)

                    avg_time = np.mean(times)
                    std_time = np.std(times)

                    results.append(
                        {
                            "algorithm": algorithm,
                            "data_size": size,
                            "avg_time": avg_time,
                            "std_time": std_time,
                            "throughput": size / avg_time,  # samples per second
                        }
                    )

                except Exception as e:
                    print(f"Error with {algorithm}, size={size}: {e}")

        return pd.DataFrame(results)

    def benchmark_recommendation_quality(self, n_trials: int = 10) -> Dict[str, Any]:
        """Benchmark the quality and consistency of top-3 recommendations."""

        results = {
            "consistency_scores": [],
            "recommendation_times": [],
            "top_algorithms": [],
            "epsilon_ranges": [],
        }

        print(f"Running {n_trials} recommendation quality tests...")

        for trial in range(n_trials):
            # Generate random test data
            data_type = np.random.choice(
                ["normal", "uniform", "exponential", "bimodal"]
            )
            size = np.random.choice([500, 1000, 2000])
            data = self.generate_test_data(data_type, size, seed=trial)

            # Time the recommendation process
            start_time = time.time()
            try:
                top3 = recommend_top3(data, n_evals=3, init_points=2, n_iter=3)
                recommendation_time = time.time() - start_time

                results["recommendation_times"].append(recommendation_time)
                results["top_algorithms"].append([rec["algorithm"] for rec in top3])
                results["epsilon_ranges"].append([rec["epsilon"] for rec in top3])

                # Calculate consistency score (how often same algorithm appears in top 3)
                if len(results["top_algorithms"]) > 1:
                    current_top = set(results["top_algorithms"][-1])
                    previous_top = set(results["top_algorithms"][-2])
                    consistency = len(current_top.intersection(previous_top)) / 3.0
                    results["consistency_scores"].append(consistency)

            except Exception as e:
                print(f"Error in trial {trial}: {e}")

        return results

    def generate_performance_report(self, benchmark_df: pd.DataFrame) -> str:
        """Generate a comprehensive performance report."""

        report = []
        report.append("PRESTO Performance Benchmark Report")
        report.append("=" * 50)
        report.append()

        # Algorithm ranking by speed
        speed_ranking = (
            benchmark_df.groupby("algorithm")["execution_time"].mean().sort_values()
        )
        report.append("Algorithm Speed Ranking (fastest to slowest):")
        for i, (algo, time) in enumerate(speed_ranking.items(), 1):
            report.append(f"{i}. {algo}: {time:.4f}s average")
        report.append()

        # Algorithm ranking by utility
        utility_ranking = (
            benchmark_df.groupby("algorithm")["utility_score"]
            .mean()
            .sort_values(ascending=False)
        )
        report.append("Algorithm Utility Ranking (best to worst):")
        for i, (algo, score) in enumerate(utility_ranking.items(), 1):
            report.append(f"{i}. {algo}: {score:.4f} average utility score")
        report.append()

        # Data type performance
        report.append("Performance by Data Type:")
        for data_type in benchmark_df["data_type"].unique():
            subset = benchmark_df[benchmark_df["data_type"] == data_type]
            best_algo = subset.groupby("algorithm")["utility_score"].mean().idxmax()
            best_score = subset.groupby("algorithm")["utility_score"].mean().max()
            report.append(f"{data_type:12s}: Best = {best_algo} ({best_score:.4f})")
        report.append()

        # Memory efficiency
        memory_ranking = (
            benchmark_df.groupby("algorithm")["memory_mb"].mean().sort_values()
        )
        report.append("Memory Efficiency (MB used):")
        for algo, memory in memory_ranking.items():
            report.append(f"{algo}: {memory:.2f} MB average")
        report.append()

        # Recommendations
        report.append("Recommendations:")
        fastest_algo = speed_ranking.index[0]
        best_utility_algo = utility_ranking.index[0]
        most_memory_efficient = memory_ranking.index[0]

        report.append(f"• For speed-critical applications: {fastest_algo}")
        report.append(f"• For maximum utility preservation: {best_utility_algo}")
        report.append(f"• For memory-constrained environments: {most_memory_efficient}")

        return "\n".join(report)

    def plot_performance_summary(
        self, benchmark_df: pd.DataFrame, save_path: str = None
    ):
        """Create comprehensive performance visualization."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "PRESTO Algorithm Performance Summary", fontsize=16, fontweight="bold"
        )

        # 1. Execution time by algorithm
        ax1 = axes[0, 0]
        speed_data = (
            benchmark_df.groupby("algorithm")["execution_time"].mean().sort_values()
        )
        speed_data.plot(kind="bar", ax=ax1, color="skyblue")
        ax1.set_title("Average Execution Time")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Utility score by algorithm
        ax2 = axes[0, 1]
        utility_data = (
            benchmark_df.groupby("algorithm")["utility_score"]
            .mean()
            .sort_values(ascending=False)
        )
        utility_data.plot(kind="bar", ax=ax2, color="lightgreen")
        ax2.set_title("Average Utility Score")
        ax2.set_ylabel("Utility Score")
        ax2.tick_params(axis="x", rotation=45)

        # 3. Memory usage
        ax3 = axes[0, 2]
        memory_data = (
            benchmark_df.groupby("algorithm")["memory_mb"].mean().sort_values()
        )
        memory_data.plot(kind="bar", ax=ax3, color="orange")
        ax3.set_title("Average Memory Usage")
        ax3.set_ylabel("Memory (MB)")
        ax3.tick_params(axis="x", rotation=45)

        # 4. Performance by data size
        ax4 = axes[1, 0]
        size_data = benchmark_df.groupby("data_size")["execution_time"].mean()
        size_data.plot(kind="line", ax=ax4, marker="o", color="red")
        ax4.set_title("Execution Time vs Data Size")
        ax4.set_xlabel("Data Size")
        ax4.set_ylabel("Time (seconds)")
        ax4.set_xscale("log")

        # 5. Utility vs Privacy (epsilon)
        ax5 = axes[1, 1]
        eps_data = benchmark_df.groupby("epsilon")["utility_score"].mean()
        eps_data.plot(kind="line", ax=ax5, marker="s", color="purple")
        ax5.set_title("Utility vs Privacy Level")
        ax5.set_xlabel("Epsilon (lower = more private)")
        ax5.set_ylabel("Utility Score")
        ax5.set_xscale("log")

        # 6. Error rates by data type
        ax6 = axes[1, 2]
        error_data = (
            benchmark_df.groupby(["data_type", "algorithm"])["mean_relative_error"]
            .mean()
            .unstack()
        )
        error_data.plot(kind="bar", ax=ax6, width=0.8)
        ax6.set_title("Mean Relative Error by Data Type")
        ax6.set_ylabel("Relative Error")
        ax6.tick_params(axis="x", rotation=45)
        ax6.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Performance plots saved to: {save_path}")

        plt.show()


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark suite."""

    print("🚀 PRESTO Comprehensive Benchmark Suite")
    print("=" * 50)

    benchmark = PRESTOBenchmark()

    # 1. Core performance benchmark
    print("1. Running core performance benchmarks...")
    performance_df = benchmark.benchmark_algorithm_performance(
        algorithms=["DP_Gaussian", "DP_Laplace", "DP_Exponential", "svt", "percentile"],
        data_sizes=[100, 500, 1000],
        data_types=["normal", "uniform", "exponential"],
        epsilon_values=[0.5, 1.0, 2.0],
    )

    # 2. Scalability benchmark
    print("\n2. Running scalability benchmarks...")
    scalability_df = benchmark.benchmark_scalability([100, 500, 1000, 2000, 5000])

    # 3. Recommendation quality benchmark
    print("\n3. Running recommendation quality benchmarks...")
    quality_results = benchmark.benchmark_recommendation_quality(n_trials=5)

    # 4. Generate report
    print("\n4. Generating performance report...")
    report = benchmark.generate_performance_report(performance_df)
    print("\n" + report)

    # 5. Create visualizations
    print("\n5. Creating performance visualizations...")
    benchmark.plot_performance_summary(performance_df, "presto_performance_summary.png")

    # 6. Save detailed results
    performance_df.to_csv("presto_benchmark_results.csv", index=False)
    scalability_df.to_csv("presto_scalability_results.csv", index=False)

    print("\n✅ Benchmark complete!")
    print("📊 Results saved to CSV files and performance plots generated")

    return performance_df, scalability_df, quality_results


if __name__ == "__main__":
    run_comprehensive_benchmark()
