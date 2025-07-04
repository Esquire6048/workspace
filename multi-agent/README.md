# 论文

## Multi-Agent Multi-View Collaborative Perception Based on Semi-Supervised Online Evolutive Learning

### 年份

2022

### 背景

很多实时场景（如自动驾驶、智慧工厂、智能安防）都需要多个传感器（摄像头、雷达等）同时感知环境，并在边缘侧（设备本地而非云端）完成快速识别和决策

每个设备（智能体、agent）之间相互协作，利用不同视角互相补充，提升整体准确性

### 核心思想

MACP（Multi-view Agent’s Collaborative Perception）

1. 多视角多智能体

* 一个场景由多个智能体（摄像头、雷达等）从不同角度观测，同一时刻可以得到多种“视角信息”。
* 每个视角对应一个识别模型。

2. 自监督初始化

* 刚开始，给每个视角的模型做自监督预训练。
* 这样每个模型在一开始就有能力提取出与自己视角相关、且和其他视角有一定“互补性”的特征。

3. 半监督一致性学习

* 对于新采集到的未标注数据：

  * 各个视角模型先独立做预测；
  * 把各个模型的预测结果**融合**，生成高置信度的伪标签；
  * 用这些伪标签做一致性正则化：即要求相同输入在数据增强或扰动下，各模型输出尽量一致。

* 这样就能充分利用未标注数据，不断优化模型。

4. 视角独立性保持

* 多模型在协同训练时容易变得相似，失去互补性。
* 为了防止这一点，MACP在训练中加入了对关键参数的限制，鼓励每个模型保持自己的“辨别独立性”，从而发挥多视角互补的优势。

## Practical Collaborative Perception: A Framework for Asynchronous and Multi-Agent 3D Object Detection

### 年份

2023

### 背景

在自动驾驶和智能交通场景中，LiDAR 3D目标检测因视角受限、遮挡严重而面临挑战，尤其是在城市交通中，当自车视野被大量道路使用者遮挡时，单车的检测可靠性急剧下降。多智能体协同感知（Vehicle-to-Everything, V2X）通过多辆车或路侧单元共享感知信息，可整合不同位置的观测，构建更完整的场景表示，从而缓解遮挡问题。

协同感知的核心难题是“性能—带宽”权衡，主要分为：
* Early Collaboration（早期融合）：各智能体直接交换原始点云并拼接输入检测器，性能最高但带宽开销巨大（通常十几MB级别）。
* Late Collaboration（后期融合）：仅交换检测结果（3D Bounding Box），带宽最低但性能提升有限，甚至在高延迟/噪声情况下不如单车检测。
* Mid Collaboration（中期融合）：交换中间特征（如BEV图）并通过GNN或Transformer等深度融合，兼顾性能和带宽，但架构复杂，且对智能体同步性、检测模型一致性有强依赖。

现有方法的主要挑战
* 架构复杂：中期融合需引入压缩/解压、自适应对齐、特征融合等多个子模块，显著改动单车检测模型架构。
* 同步假设不现实：大多数方法假设所有智能体严格同步采集与处理点云，或交换的BEV图具有相同时间戳，实际中仅能保证GPS时间参考一致而无完美同步。
* 检测模型异构支持差：不同厂商或版本的检测器生成的BEV图存在分布差异，需额外模块弥合域间差距，进一步增加复杂度。

### 核心思想

Late-Early Collaboration

作者提出一种简单高效的协同策略，取用后期融合（交换检测结果）的带宽优势，同时借鉴早期融合（拼接点云）的性能提升：
* 信息交换：各智能体仅广播自身检测到的3D目标框和对应速度预测；
* 异步处理：只要求统一GPS时间参考，不强制同步采集，接收的检测结果可带有时差；
* 时序对齐：利用点云序列中的场景流（scene flow）模块，将过去时刻检测到的目标位置按速度推算到当前时刻，实现时空对齐；
* 融合输入：将对齐后的目标框对应的“伪点云”（由MoDAR方法将检测框转为点＋附加特征）拼接到自车当前点云中，再输入原有单车检测模型，无需改动模型架构核心。

该策略被称为晚早协同（late-early collaboration），兼顾后期与早期融合优点，同时满足：
* 极低带宽：仅交换3D框与少量速度信息，平均约0.01 MB／智能体；
* 最小改动：无需新增复杂融合网络，仅插入场景流、MoDAR等轻量插件模块；
* 弱同步假设：仅需GPS时间参考，容忍各智能体检测结果时差；
* 异构检测器支持：不同检测器输出的3D框格式统一，无域差异问题。

## Learning Distilled Collaboration Graph for Multi-Agent Perception

### 年份



### 背景



### 核心思想



## 

### 年份



### 背景



### 核心思想



