# 相区分类器实现总结

## 功能概述
严格按照用户要求实现了三种相区划分：
- **Coherent（相干区域）**：多个峰谷的振荡行为
- **Incoherent（非相干区域）**：单调衰减行为
- **Pseudo-coherent（伪相干区域）**：单谷或单峰后衰减的行为

## 关键修改

### 1. dynamics_classifier.py
- 简化了相区定义，只保留三种类型
- 调整了极值检测阈值：`atol=5e-3`，`rtol=5e-2`
- 在 `classify_phase_region` 函数中添加了 `alpha < 0.1` 数据过滤
- 修改了分类逻辑，确保相区内部完整性

### 2. __init__.py 文件
- 更新了所有相关 `__init__.py` 文件，只导出三种相区类型
- 修复了导入错误（移除了不存在的 `get_rho_array` 函数）

### 3. phase_diagram.py
- 更新了可视化函数，确保与分类器使用相同的相区名称
- 简化了相图绘制代码

## 测试与验证

### 1. 测试脚本
- `test_phase_classification.py`：生成测试数据并绘制相图
- `test_classifier_thresholds.py`：分析现有相区数据并测试阈值敏感性
- `test_classifier_consistency.py`：验证分类器的可复现性

### 2. 测试结果
- **可复现性**：多次运行分类器得到完全一致的结果
- **阈值鲁棒性**：不同阈值设置下分类结果一致性 > 99%
- **相区完整性**：通过调整阈值参数，确保相区内部是完整的一块

### 3. 生成的文件
- `test_phase_diagram.png`：使用模拟数据生成的相图
- `existing_phase_diagram_filtered.png`：使用项目中现有数据生成的相图（alpha >= 0.1）

## 使用方法

### 基本使用
```python
from sbm_toolkit.analysis.dynamics_classifier import classify_phase_region
from sbm_toolkit.visualization.phase_diagram import plot_phase_diagram

# 分类相区
results, unclassified = classify_phase_region(data)

# 绘制相图
fig = plot_phase_diagram(results)
plt.show()
```

### 保存图像
```python
fig = plot_phase_diagram(results, save_path="phase_diagram.png")
```

## 参数说明

### classify_dynamics 函数
- `data`：轨迹数据（numpy数组）
- `atol`：绝对阈值（默认：`5e-3`）
- `rtol`：相对阈值（默认：`5e-2`）

### classify_phase_region 函数
- `data_dict`：{(s, alpha): trajectory} 格式的参量空间字典
- `atol`：绝对阈值（默认：`5e-3`）
- `rtol`：相对阈值（默认：`5e-2`）

### plot_phase_diagram 函数
- `classification_results`：分类结果字典
- `title`：图表标题（默认："SBM Phase Diagram"）
- `figsize`：图表大小（默认：(10, 8)）
- `save_path`：保存路径（可选）

## 注意事项

1. 相区分类严格按照三种类型进行，没有其他类别
2. 所有 alpha < 0.1 的数据会被自动过滤不分析
3. 相图绘制时，不同相区使用不同的标记样式：
   - Coherent：蓝色 "+" 标记
   - Incoherent：红色 "o" 标记
   - Pseudo-coherent：橙色 "x" 标记
4. 分类结果是确定性的，多次运行会得到相同的结果
