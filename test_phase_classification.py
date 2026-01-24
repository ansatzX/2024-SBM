#!/usr/bin/env python3
"""
测试相区分类功能的脚本

运行方法: python3 test_phase_classification.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sbm_toolkit.analysis.dynamics_classifier import (
    classify_phase_region,
    DYNAMICS_COHERENT,
    DYNAMICS_INCOHERENT,
    DYNAMICS_PSEUDO_COHERENT,
    classify_dynamics
)
from sbm_toolkit.visualization.phase_diagram import plot_phase_diagram


def generate_test_data(n_points: int = 100) -> dict:
    """
    生成测试数据，模拟不同区域的行为

    Args:
        n_points: 点数

    Returns:
        数据字典
    """
    data = {}

    # 生成 incoherent 区域的数据 (alpha 较大或 s 较小)
    for i in range(n_points // 3):
        s = np.random.uniform(0.0, 0.5)
        alpha = np.random.uniform(0.2, 0.8)
        # 生成单调递减的数据
        trajectory = np.exp(-0.05 * np.arange(100)) + np.random.normal(0, 0.02, 100)
        data[(s, alpha)] = trajectory

    # 生成 coherent 区域的数据
    for i in range(n_points // 3, 2 * n_points // 3):
        s = np.random.uniform(0.3, 0.8)
        alpha = np.random.uniform(0.15, 0.6)
        # 生成振荡衰减数据
        trajectory = np.exp(-0.02 * np.arange(100)) * np.sin(0.5 * np.arange(100)) + 0.5
        data[(s, alpha)] = trajectory

    # 生成 pseudo-coherent 区域的数据
    for i in range(2 * n_points // 3, n_points):
        s = np.random.uniform(0.6, 0.95)
        alpha = np.random.uniform(0.15, 0.5)
        # 生成单谷数据
        trajectory = 0.5 + 0.3 * np.exp(-0.05 * np.arange(100)) * np.sin(0.3 * np.arange(100))
        data[(s, alpha)] = trajectory

    return data


def test_phase_classification():
    """测试相区分类功能"""
    print("=== 生成测试数据 ===")
    data = generate_test_data(300)

    print("=== 分类相区 ===")
    results, unclassified = classify_phase_region(data)
    print(f"分类结果点数: {len(results)}")
    print(f"未分类点数: {len(unclassified)}")

    # 统计各相区数量
    phase_counts = {
        DYNAMICS_COHERENT: 0,
        DYNAMICS_INCOHERENT: 0,
        DYNAMICS_PSEUDO_COHERENT: 0
    }
    for phase in results.values():
        phase_counts[phase] += 1

    print("\n=== 相区统计 ===")
    print(f"  相干区域 (coherent): {phase_counts[DYNAMICS_COHERENT]}")
    print(f"  非相干区域 (incoherent): {phase_counts[DYNAMICS_INCOHERENT]}")
    print(f"  伪相干区域 (pseudo-coherent): {phase_counts[DYNAMICS_PSEUDO_COHERENT]}")

    # 绘制相图
    print("\n=== 绘制相图 ===")
    fig = plot_phase_diagram(results,
                           title="SBM 相图 (测试数据)",
                           save_path="test_phase_diagram.png")
    plt.show()


if __name__ == "__main__":
    test_phase_classification()
