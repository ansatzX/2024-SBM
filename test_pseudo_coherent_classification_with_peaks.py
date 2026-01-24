#!/usr/bin/env python3
"""
测试伪相干区域分类的简单脚本，打印峰谷检测结果
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sbm_toolkit.analysis.dynamics_classifier import (
    classify_dynamics,
    detect_monotonic_segments,
    DYNAMICS_COHERENT,
    DYNAMICS_INCOHERENT,
    DYNAMICS_PSEUDO_COHERENT,
)


def test_pseudo_coherent_classification():
    """测试伪相干区域的分类"""
    print("=== 测试伪相干区域分类 ===")

    # 创建一个简单的单谷后衰减的轨迹
    t = np.arange(100)
    # 先下降到最小值，然后逐渐上升并趋于平稳
    trajectory = 0.5 + 0.3 * np.exp(-0.03 * t) * np.sin(0.1 * t)
    trajectory += np.random.normal(0, 0.01, 100)  # 添加噪声

    # 检测峰和谷
    max_indices = signal.argrelmax(trajectory)[0]
    min_indices = signal.argrelmin(trajectory)[0]

    # 检测单调段
    segments = detect_monotonic_segments(trajectory.tolist())

    print(f"峰的数量: {len(max_indices)}")
    print(f"谷的数量: {len(min_indices)}")
    print(f"单调段类型: {segments}")

    # 分类轨迹
    classification = classify_dynamics(trajectory)
    print(f"分类结果: {classification}")

    # 绘制轨迹
    plt.figure(figsize=(10, 6))
    plt.plot(t, trajectory, label="Test Trajectory")
    for i in max_indices:
        plt.scatter(i, trajectory[i], color='red', label='Peak')
    for i in min_indices:
        plt.scatter(i, trajectory[i], color='blue', label='Valley')
    plt.xlabel("Time")
    plt.ylabel("Spin Population")
    plt.title(f"Trajectory Classification: {classification}")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_pseudo_coherent_trajectory_with_peaks.png")
    print("轨迹图像已保存到 test_pseudo_coherent_trajectory_with_peaks.png")

    return classification


def test_pseudo_coherent_with_single_valley():
    """测试只有一个谷的轨迹"""
    print("\n=== 测试只有一个谷的轨迹 ===")

    t = np.arange(100)
    trajectory = 0.6 - 0.2 * np.exp(-0.02 * t) * np.cos(0.1 * t)
    trajectory += np.random.normal(0, 0.01, 100)

    max_indices = signal.argrelmax(trajectory)[0]
    min_indices = signal.argrelmin(trajectory)[0]

    print(f"峰的数量: {len(max_indices)}")
    print(f"谷的数量: {len(min_indices)}")

    classification = classify_dynamics(trajectory)
    print(f"分类结果: {classification}")

    plt.figure(figsize=(10, 6))
    plt.plot(t, trajectory, label="Single Valley Trajectory")
    for i in max_indices:
        plt.scatter(i, trajectory[i], color='red', label='Peak')
    for i in min_indices:
        plt.scatter(i, trajectory[i], color='blue', label='Valley')
    plt.xlabel("Time")
    plt.ylabel("Spin Population")
    plt.title(f"Trajectory Classification: {classification}")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_single_valley_trajectory_with_peaks.png")
    print("轨迹图像已保存到 test_single_valley_trajectory_with_peaks.png")

    return classification


def test_pseudo_coherent_with_single_peak_and_valley():
    """测试有一个峰和一个谷的轨迹"""
    print("\n=== 测试有一个峰和一个谷的轨迹 ===")

    t = np.arange(100)
    trajectory = 0.5 + 0.3 * np.exp(-0.02 * t) * np.sin(0.15 * t)
    trajectory += np.random.normal(0, 0.01, 100)

    max_indices = signal.argrelmax(trajectory)[0]
    min_indices = signal.argrelmin(trajectory)[0]

    print(f"峰的数量: {len(max_indices)}")
    print(f"谷的数量: {len(min_indices)}")

    classification = classify_dynamics(trajectory)
    print(f"分类结果: {classification}")

    plt.figure(figsize=(10, 6))
    plt.plot(t, trajectory, label="Single Peak and Valley Trajectory")
    for i in max_indices:
        plt.scatter(i, trajectory[i], color='red', label='Peak')
    for i in min_indices:
        plt.scatter(i, trajectory[i], color='blue', label='Valley')
    plt.xlabel("Time")
    plt.ylabel("Spin Population")
    plt.title(f"Trajectory Classification: {classification}")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_peak_valley_trajectory_with_peaks.png")
    print("轨迹图像已保存到 test_peak_valley_trajectory_with_peaks.png")

    return classification


if __name__ == "__main__":
    test_pseudo_coherent_classification()
    test_pseudo_coherent_with_single_valley()
    test_pseudo_coherent_with_single_peak_and_valley()
    print("\n=== 测试完成 ===")