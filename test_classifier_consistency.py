#!/usr/bin/env python3
"""
测试相区分类器的一致性和可复现性

这个脚本会：
1. 使用固定的随机种子生成相同的测试数据
2. 多次运行分类器，检查结果是否一致
3. 测试不同参数组合的分类稳定性
"""

import numpy as np
from sbm_toolkit.analysis.dynamics_classifier import (
    classify_phase_region,
    classify_dynamics,
    DYNAMICS_COHERENT,
    DYNAMICS_INCOHERENT,
    DYNAMICS_PSEUDO_COHERENT,
)


def generate_reproducible_test_data(n_points=100, seed=42):
    """
    生成可复现的测试数据

    Args:
        n_points: 数据点数量
        seed: 随机种子

    Returns:
        测试数据字典
    """
    np.random.seed(seed)
    data = {}

    # 生成相干区域数据 (s 适中，alpha 适中)
    for i in range(n_points // 3):
        s = np.random.uniform(0.3, 0.7)
        alpha = np.random.uniform(0.2, 0.6)
        t = np.arange(100)
        # 阻尼振荡轨迹
        trajectory = 0.5 + 0.4 * np.exp(-0.02 * t) * np.sin(0.5 * t) + np.random.normal(0, 0.03, 100)
        data[(s, alpha)] = trajectory

    # 生成非相干区域数据 (s 小，alpha 大)
    for i in range(n_points // 3, 2 * n_points // 3):
        s = np.random.uniform(0.05, 0.3)
        alpha = np.random.uniform(0.5, 0.95)
        t = np.arange(100)
        # 单调递减轨迹
        trajectory = 0.5 + 0.4 * np.exp(-0.05 * t) + np.random.normal(0, 0.03, 100)
        data[(s, alpha)] = trajectory

    # 生成伪相干区域数据 (s 大，alpha 小)
    for i in range(2 * n_points // 3, n_points):
        s = np.random.uniform(0.7, 0.95)
        alpha = np.random.uniform(0.1, 0.3)
        t = np.arange(100)
        # 单谷轨迹
        trajectory = 0.5 + 0.4 * np.exp(-0.03 * t) * np.sin(0.2 * t) + np.random.normal(0, 0.03, 100)
        data[(s, alpha)] = trajectory

    return data


def test_classification_reproducibility():
    """测试分类器的可复现性"""
    print("=== 测试分类器的可复现性 ===")
    test_data = generate_reproducible_test_data()

    # 多次运行分类器
    n_runs = 5
    results_list = []

    for i in range(n_runs):
        results, unclassified = classify_phase_region(test_data)
        results_list.append(results)

        print(f"Run {i+1}: {len(results)} points classified")
        if unclassified:
            print(f"  Unclassified: {len(unclassified)} points")

        # 统计各相区数量
        coherent_count = sum(1 for r in results.values() if r == DYNAMICS_COHERENT)
        incoherent_count = sum(1 for r in results.values() if r == DYNAMICS_INCOHERENT)
        pseudo_count = sum(1 for r in results.values() if r == DYNAMICS_PSEUDO_COHERENT)
        print(f"  Coherent: {coherent_count}, Incoherent: {incoherent_count}, Pseudo-coherent: {pseudo_count}")

    # 检查所有运行的结果是否一致
    all_consistent = True
    first_results = results_list[0]
    for i, run_results in enumerate(results_list[1:], start=2):
        if run_results != first_results:
            print(f"Run {i} differs from Run 1")
            # 找到差异点
            differing_points = []
            for key in first_results.keys():
                if first_results[key] != run_results[key]:
                    differing_points.append((key, first_results[key], run_results[key]))
            print(f"  Differences: {len(differing_points)} points")
            for point, run1, run2 in differing_points[:5]:
                print(f"    (s={point[0]:.3f}, α={point[1]:.3f}): {run1} → {run2}")
            all_consistent = False

    if all_consistent:
        print("✅ 所有运行的结果一致")
    else:
        print("❌ 运行结果不一致")

    return all_consistent


def test_threshold_robustness():
    """测试不同阈值对分类的影响"""
    print("\n=== 测试阈值鲁棒性 ===")
    test_data = generate_reproducible_test_data(n_points=50)

    thresholds = [
        (1e-3, 1e-2),  # 严格
        (5e-3, 5e-2),  # 适中（默认）
        (1e-2, 1e-1),  # 宽松
    ]

    classifications = []

    for i, (atol, rtol) in enumerate(thresholds):
        results, unclassified = classify_phase_region(test_data, atol, rtol)
        classifications.append(results)

        print(f"Threshold {i+1}: atol={atol:.1e}, rtol={rtol:.1e}")
        print(f"  Points classified: {len(results)}, Unclassified: {len(unclassified)}")
        coherent_count = sum(1 for r in results.values() if r == DYNAMICS_COHERENT)
        incoherent_count = sum(1 for r in results.values() if r == DYNAMICS_INCOHERENT)
        pseudo_count = sum(1 for r in results.values() if r == DYNAMICS_PSEUDO_COHERENT)
        print(f"  Coherent: {coherent_count}, Incoherent: {incoherent_count}, Pseudo-coherent: {pseudo_count}")

    # 计算一致性
    main_classification = classifications[1]  # 使用默认阈值
    for i, other_classification in enumerate(classifications):
        if i == 1:
            continue

        differing_count = 0
        for key in main_classification.keys():
            if main_classification[key] != other_classification.get(key, "unclassified"):
                differing_count += 1

        consistency = (len(main_classification) - differing_count) / len(main_classification) * 100
        print(f"\n一致性 with threshold {i+1}: {consistency:.1f}%")


def test_single_trajectory_consistency():
    """测试单个轨迹的分类一致性"""
    print("\n=== 测试单个轨迹的分类一致性 ===")

    np.random.seed(42)
    t = np.arange(100)
    trajectory = 0.5 + 0.4 * np.exp(-0.02 * t) * np.sin(0.5 * t) + np.random.normal(0, 0.03, 100)

    classification1 = classify_dynamics(trajectory)
    classification2 = classify_dynamics(trajectory)

    print(f"分类1: {classification1}")
    print(f"分类2: {classification2}")

    if classification1 == classification2:
        print("✅ 相同轨迹分类一致")
    else:
        print("❌ 相同轨迹分类不一致")


if __name__ == "__main__":
    print("相区分类器一致性测试")
    print("=" * 40)

    # 运行所有测试
    reproducibility_passed = test_classification_reproducibility()
    test_threshold_robustness()
    test_single_trajectory_consistency()

    print("\n" + "=" * 40)
    if reproducibility_passed:
        print("✅ 所有一致性测试通过")
    else:
        print("❌ 部分一致性测试未通过")
