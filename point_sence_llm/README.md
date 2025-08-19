## 数据集

### 有损 ModelNet40

https://github.com/jiachens/ModelNet40-C

### 室内场景 Replica

https://github.com/facebookresearch/Replica-Dataset

### 视觉-语言对 室内场景 SceneVerse

https://github.com/scene-verse/SceneVerse

## 工作

### 3D-LLM: Injecting the 3D World into LLMs

https://github.com/UMass-Embodied-AGI/3D-LLM

核心想法：把“3D世界”注入到现有LLM里，让LLM能处理点云/场景层面的任务（caption、dense caption、3D-VQA、3D grounding、任务分解、导航等）。

怎么接入3D：不用从零训练3D编码器，而是把多视角渲染图的2D语义特征“重建”为3D特征，再喂给以2D-VLM为骨干的模型；同时引入3D定位机制（位置嵌入+位置token）增强空间理解。

数据与训练：通过三类提示流程自动构造约100万条3D-语言数据；主干沿用BLIP-2/Flamingo等2D-VLM以提高训练效率。

结果：在 ScanQA/SQA3D/3DMV-VQA 等基准超过SOTA，ScanQA 上 BLEU-1 提升约9%。

局限：依赖“将3D场景渲染成多视图图像”的过程，带来渲染/投影流水线成本。

### LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning & Planning

https://github.com/Open3DA/LL3DA

核心想法：直接以点云为输入（非多视图特征），同时支持视觉交互提示（点击、框选等）以消解文本歧义，面向理解-推理-规划的一体化3D助手。

怎么接入3D：提出 Interactor3D 多模态 Transformer，将文本指令、视觉提示与3D场景编码成固定长度的可学习查询token，再作为前缀送入冻结的LLM；用位置/ROI嵌入把点击/框坐标显式注入。

结果：在3D密集描述与3D-VQA等任务上超过多种3D-VL模型；可通过交互去除复杂场景中的歧义。

### Chat-3D: Data-efficiently Tuning LLM for Universal Dialogue of 3D Scenes

https://github.com/Chat-3D/Chat-3D

核心想法：将预训练3D表示对齐到LLM语义空间，打造“通用3D场景对话”系统，强调数据效率。

训练策略：提出三阶段对齐策略，并构建高质量“以对象为中心”的3D指令数据集与相应提示模板以增强推理与交互体验。

结果：在其构建的数据集上相对 GPT-4 达到 75.6% 的得分。

