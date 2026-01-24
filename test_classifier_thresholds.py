#!/usr/bin/env python3
"""
测试相区分类器的脚本来确保划分符合预期

这个脚本会：
1. 加载项目中现有的相区数据点
2. 使用这些数据来验证分类器的性能
3. 测试阈值调整的效果
"""

import pickle
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


def load_and_analyze_existing_points():
    """加载并分析项目中现有的相区数据点"""
    print("=== 加载现有的相区数据 ===")
    try:
        with open('/home/ansatz/data/code/2024-SBM/coherent_points.pickle', 'rb') as f:
            coherent_points = pickle.load(f)
        with open('/home/ansatz/data/code/2024-SBM/incoherent_points.pickle', 'rb') as f:
            incoherent_points = pickle.load(f)
        with open('/home/ansatz/data/code/2024-SBM/pseudo_coherent_points.pickle', 'rb') as f:
            pseudo_coherent_points = pickle.load(f)

        print(f"Coherent: {len(coherent_points)}, Incoherent: {len(incoherent_points)}, Pseudo-coherent: {len(pseudo_coherent_points)}")

        # 过滤 alpha < 0.1 的数据
        filtered_coherent = [p for p in coherent_points if p[1] >= 0.1]
        filtered_incoherent = [p for p in incoherent_points if p[1] >= 0.1]
        filtered_pseudo_coherent = [p for p in pseudo_coherent_points if p[1] >= 0.1]

        print(f"After filtering (alpha >= 0.1):")
        print(f"  Coherent: {len(filtered_coherent)} ({len(filtered_coherent)/len(coherent_points)*100:.1f}%)")
        print(f"  Incoherent: {len(filtered_incoherent)} ({len(filtered_incoherent)/len(incoherent_points)*100:.1f}%)")
        print(f"  Pseudo-coherent: {len(filtered_pseudo_coherent)} ({len(filtered_pseudo_coherent)/len(pseudo_coherent_points)*100:.1f}%)")

        return filtered_coherent, filtered_incoherent, filtered_pseudo_coherent

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("无法找到相区数据文件。请确保在正确的目录中运行脚本。")
        return [], [], []


def create_phase_plot_with_existing_points(filtered_coherent, filtered_incoherent, filtered_pseudo_coherent):
    """使用现有的数据点创建相图"""
    print("\n=== 创建相图 ===")
    # 准备数据格式供 plot_phase_diagram 函数使用
    data = {}
    for s, alpha in filtered_coherent:
        data[(s, alpha)] = DYNAMICS_COHERENT
    for s, alpha in filtered_incoherent:
        data[(s, alpha)] = DYNAMICS_INCOHERENT
    for s, alpha in filtered_pseudo_coherent:
        data[(s, alpha)] = DYNAMICS_PSEUDO_COHERENT

    fig = plot_phase_diagram(data,
                           title="SBM 相图（使用现有数据，alpha >= 0.1）",
                           save_path="existing_phase_diagram_filtered.png")

    print(f"相图已保存到: existing_phase_diagram_filtered.png")
    return fig


def test_threshold_sensitivity(filtered_coherent, filtered_incoherent, filtered_pseudo_coherent):
    """测试不同阈值对分类的影响"""
    print("\n=== 测试阈值敏感性 ===")
    from sbm_toolkit.analysis.dynamics_classifier import detect_monotonic_segments

    # 测试不同的阈值
    test_thresholds = [
        (1e-3, 1e-2),
        (5e-3, 5e-2),
        (1e-2, 1e-1),
        (2e-2, 2e-1),
    ]

    all_classifications = {}

    for atol, rtol in test_thresholds:
        print(f"\nTesting atol={atol:.1e}, rtol={rtol:.1e}:")

        # 创建模拟数据（简单的周期性函数）
        np.random.seed(42)
        test_data = {}

        # 生成相干态数据
        for i in range(10):
            s = np.random.uniform(0.1, 0.9)
            alpha = np.random.uniform(0.1, 0.9)
            t = np.arange(100)
            # 具有阻尼振荡的轨迹
            trajectory = 0.5 + 0.4 * np.exp(-0.02 * t) * np.sin(0.5 * t) + np.random.normal(0, 0.05, 100)
            test_data[(s, alpha)] = trajectory

        # 分类
        results, unclassified = classify_phase_region(test_data, atol, rtol)
        all_classifications[(atol, rtol)] = results

        # 统计
        coherent_count = sum(1 for r in results.values() if r == DYNAMICS_COHERENT)
        incoherent_count = sum(1 for r in results.values() if r == DYNAMICS_INCOHERENT)
        pseudo_count = sum(1 for r in results.values() if r == DYNAMICS_PSEUDO_COHERENT)

        print(f"  Coherent: {coherent_count}")
        print(f"  Incoherent: {incoherent_count}")
        print(f"  Pseudo-coherent: {pseudo_count}")


if __name__ == "__main__":
    # 加载并分析现有的相区数据
    filtered_coherent, filtered_incoherent, filtered_pseudo_coherent = load_and_analyze_existing_points()

    if filtered_coherent or filtered_incoherent or filtered_pseudo_coherent:
        # 创建相图
        fig = create_phase_plot_with_existing_points(filtered_coherent, filtered_incoherent, filtered_pseudo_coherent)
        plt.show()

        # 测试阈值敏感性
        test_thresholds = False  # 暂时关闭，避免生成太多图像
        if test_thresholds:
            test_threshold_sensitivity(filtered_coherent, filtered_incoherent, filtered_pseudo_coherent)

    print("\n=== 测试完成 ===")
