# 四旋翼齐次控制器 — C++ 版本

C++ 实现位于 `hcs_toolbox_cpp` 工具箱中。

## 源文件

```
hcs_toolbox_cpp/src/demo_uav_homogeneous_control.cpp
```

## 构建与运行

```bash
cd /home/wzy/Homogeneous_control/hcs_toolbox_cpp/build
cmake .. && make -j$(nproc)

# 运行仿真
./bin/demo_uav_homogeneous_control

# 绘图
python3 ../scripts/plot_uav_comparison.py
```

## 输出文件

| 文件 | 内容 |
|------|------|
| `build/uav_z_comparison_cpp.csv` | Z 轴 HPIC vs 线性 PI 对比 |
| `build/uav_yaw_comparison_cpp.csv` | Yaw 轴 HPC vs 线性 PD 对比 |
| `build/uav_xy_comparison_cpp.csv` | XY 轴 HPC vs LQR 对比 |
| `scripts/uav_comparison_cpp.png` | 3×3 对比图表 |

## C++ 与 Python 结果对照

| 回路 | C++ | Python | 一致 |
|------|-----|--------|------|
| Z (HPIC) | 0.004325 | 0.004325 | ✅ |
| Yaw (HPC) | 0.000000 | 0.000000 | ✅ |
| XY (HPC) | 0.000000 | 0.000000 | ✅ |
