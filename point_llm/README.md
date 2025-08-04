### Benchmarking Robustness of 3D Point Cloud Recognition against Common Corruptions
有损的ModelNet40

#### 论文链接
https://arxiv.org/abs/2201.12296

#### Github主页
https://github.com/jiachens/ModelNet40-C

### ModelNet40
ModelNet40

#### 项目主页
https://modelnet.cs.princeton.edu/

#### 论文链接
https://arxiv.org/abs/1406.5670

#### Github主页
https://github.com/zhirongw/3DShapeNets

### ScanObjectNN
ScanObjectNN

#### 项目主页
https://hkust-vgd.github.io/scanobjectnn/

#### 论文链接
https://arxiv.org/abs/1908.04616

#### Github主页
https://github.com/hkust-vgd/scanobjectnn

## 3D物体

## MiniGPT-3D: Efficiently Aligning 3D Point Clouds with Large Language Models using 2D Priors

https://github.com/TangYuan96/MiniGPT-3D

## PointLLM: Empowering Large Language Models to Understand Point Clouds

https://github.com/InternRobotics/PointLLM

### **ShapeLLM: Universal 3D Object Understanding for Embodied Interaction**

* **输入：** 单个3D物体的点云（可带颜色）
* **输出任务：** 对物体的多层次理解与交互，如：功能分析、问答、操作建议等
* **方法：**

  * 基于ReCon++点云编码器提取几何/纹理特征；
  * 结合LLM通过700K点云-文本对预训练，+30K指令微调数据集；
  * 模型支持类ChatGPT交互，如“这是什么？有什么用？如何使用它？”
  * 在新构建的3D多模态评估集（3D MM-Vet）中取得领先。

### **PointCLIP: Point Cloud Understanding by CLIP**

* **输入：** 单物体的3D点云
* **输出任务：** 零样本3D物体识别（open-vocab分类），也支持少量学习提升性能
* **方法：**

  * 将点云投影为多个深度图（多视角），输入到CLIP的图像编码器；
  * 使用CLIP的文本编码器将候选类别转换为嵌入，与图像编码对齐；
  * 设计轻量的“inter-view adapter”模块，实现多视角特征融合；
  * 在ModelNet40等任务中零样本下达到甚至优于有监督模型的性能。

### **PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning**

* **输入：** 多类物体的3D点云（场景或单物体）
* **输出任务：**

  * 开集3D物体识别（零样本）
  * 3D部件分割（part segmentation）
  * 3D目标检测（无需训练）
* **方法：**

  * 用GPT生成富含语义的文本prompt，引导CLIP更好对齐语言和视觉特征；
  * 引入“形状投影模块”提升点云到深度图投影效果；
  * 支持无需3D标签，仅凭2D预训练模型即可完成复杂3D任务；
  * 提升原始PointCLIP准确率约40%，实现对3D对象的更深层开放式理解。

### **PointLLM: Empowering Large Language Models to Understand Point Clouds**

* **输入：** 单物体的带颜色点云
* **输出任务：**

  * 3D对象描述（caption）
  * 开放式3D问答（如“这是什么？用来干什么？”）
  * 生成式3D分类任务（输出为语言）
* **方法：**

  * 采用点云编码器提取语义+几何特征；
  * 两阶段训练：阶段1对齐点云-语言嵌入空间（对比学习）；阶段2指令微调（70K指令样本）；
  * 支持自然语言对话，模拟真实人类理解；
  * 在人类评估中超越一半以上真实人类描述者，在3D语言表达方面表现出色。

### **GPT4Point: A Unified Framework for Point-Language Understanding and Generation**

* **输入：** 3D点云（主要为单个物体，如Objaverse-XL数据集中的对象）
* **输出任务：** 双向任务

  * 理解任务：3D点云 → 文本（如描述、问答）
  * 生成任务：文本 → 3D点云（生成新形状或重建低质量点云）
* **方法：**

  * 构建Pyramid-XL大规模点云-文本数据集（超100万个对象）；
  * 使用点云编码器提取点特征，通过多层嵌入与语言模型统一；
  * 模型可进行双向推理（点到文、文到点），具备生成能力；
  * 在3D caption和QA任务上超越现有模型，同时支持条件点云生成，推动3D生成方向发展。

### **MiniGPT-3D: Efficiently Aligning 3D Point Clouds with LLMs using 2D Priors**

* **输入：** 3D点云（物体级）
* **输出任务：**

  * 3D物体分类（如ModelNet40）
  * 物体描述（caption）
* **方法：**

  * 通过深度图将3D点云转换为图像视图，并重用2D预训练的视觉语言模型（如LLaVA）；
  * 提出四阶段对齐流程，将点云特征逐步映射到语言空间；
  * 使用参数高效的训练方法（如LoRA和norm tuning），仅需约4800万可训练参数；
  * 模型在3D分类与描述任务上达成SOTA，单GPU训练时间仅27小时，效率极高。

## 相关

1. 3d-llm: Injecting the 3d world into large language models
2. HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding
3. LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities
4. Agent3D-Zero: An Agent for Zero-shot 3D Understanding
5. ShapeLLM: Universal 3D Object Understanding for Embodied Interaction
6. Scene-LLM: Extending Language Model for 3D Visual Understanding and Reasoning
7. PointCLIP: Point Cloud Understanding by CLIP
8. PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning
9. PointLLM: Empowering Large Language Models to Understand Point Clouds
10. Chat-scene: Bridging 3d scene and large language models with object identifiers
11. GPT4Point: A Unified Framework for Point-Language Understanding and Generation
12. Grounded 3D-LLM with Referent Tokens
13. LiDAR-LLM: Exploring the Potential of Large Language Models for 3D LiDAR Understanding
14. MiniGPT-3D: Efficiently Aligning 3D Point Clouds with Large Language Models using 2D Priors
15. GreenPLM: Cross-Lingual Transfer of Monolingual Pre-Trained Language Models at Almost No Cost
16. SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding
17. LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning
18. Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes
19. Visual Programming for Zero-shot Open-Vocabulary 3D Visual Grounding
20. 3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer
21. ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities
