# 3DLLM论文整理

## 3D场景

### **Agent3D-Zero: An Agent for Zero-shot 3D Understanding**

* **输入：** 3D场景，通过agent从多个视角获取图像
* **输出任务：** 零样本3D理解（无3D训练数据），包括场景问答、区域分割等
* **方法：**

  * 利用预训练VLM（如BLIP）作为观察器；
  * 模拟agent探索视角并动态构建场景理解；
  * 起始使用鸟瞰图，依赖视觉prompt引导VLM生成描述；
  * 多视角策略提高理解质量，训练0样本下实现跨场景泛化。

### **3D-LLM: Injecting the 3D World into Large Language Models**

* **输入：** 3D点云场景（如ScanNet室内场景）
* **输出任务：** 场景描述、密集描述、3D问答、任务拆解、三维指代解析、3D辅助对话、导航指导等。
* **方法：**

  * 将点云渲染为多视角图像，通过CLIP等2D图像-语言模型提取视觉特征；
  * 引入3D定位模块理解空间关系；
  * 构建覆盖多任务的30万条3D语言指令数据集；
  * 采用指令微调方式训练3D-LLM，模型在ScanQA和ScanRefer等基准上达成SOTA效果。

### **Chat-Scene: Bridging 3D Scene and Large Language Models with Object Identifiers**

* **输入：** 完整3D场景（如ScanNet），将场景表示为“对象集合”
* **输出任务：**

  * 对3D场景的问答与对话
  * 对象指代解析（object grounding）
  * 场景描述、空间推理等
* **方法：**

  * 场景中每个对象被赋予唯一标识符（如Object#5），以便语言模型识别与引用；
  * 使用统一语言格式处理多任务，如“该红色物体是什么” → 预测Object#X；
  * 使用统一QA格式微调，避免单独设计多个任务头；
  * 在ScanRefer、ScanQA、Scan2Cap等多个3D语言任务中取得领先。

### **Scene-LLM: Extending Language Model for 3D Visual Understanding and Reasoning**

* **输入：** 室内3D场景，使用融合“全局视图（global）”与“第一人称视角（egocentric）”的混合表示形式
* **输出任务：** 密集场景描述、3D问答、交互式导航与计划生成等
* **方法：**

  * 场景表示包括两个通路：全局地图（提供空间布局）与局部观察视角（提供精细语义信息）；
  * 使用投影层将3D特征映射到LLM语言空间；
  * 融合两个视图可提升小物体识别与长距离依赖建模能力；
  * 支持动态场景变化下的交互问答与计划生成；
  * 在多个3D任务上验证其统一架构的有效性。

### **Grounded 3D-LLM with Referent Tokens**

* **输入：** 3D场景 + 含有“指代token”（如<ref-chair>）的文本
* **输出任务：**

  * 统一形式的3D描述、QA、指代解析等任务
  * 模型在文本中生成带有<ref-token>的输出，并实现语义与空间绑定
* **方法：**

  * 构建包含超过100万短语与3D区域绑定的数据集；
  * 引入CLASP（对比语言-场景预训练）机制对齐referent tokens与真实区域；
  * 所有任务格式化为指令式问答，例如：“请描述场景中靠墙的物体” → “<ref-sofa>是一张灰色沙发…”；
  * 实现3D任务统一建模，提升多个任务（ScanQA、ScanRefer等）性能。

### **SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding**

* **输入：** 大规模室内3D场景（6.8万个）+ 每个场景对应的文本描述（人工+合成）
* **输出任务：**

  * 场景描述（captioning）
  * 3D问答
  * 3D视觉指代定位（grounding）
* **方法：**

  * 构建SceneVerse数据集：约250万个文本描述，涵盖复杂场景；
  * 文本部分由人工标注与自动场景图生成模型共同提供，提升多样性；
  * 提出“GPS预训练策略”（Grounded Pretraining for Scenes）统一训练模型，覆盖多个任务目标；
  * 在ScanRefer等多个基准任务上取得SOTA；
  * 支持零样本迁移，即预训练后无需微调可直接迁移至新任务。

### **LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning**

* **输入：** 点云表示的3D场景 + 用户视觉交互（如点击、选中区域）+ 文本问题或指令
* **输出任务：**

  * 密集3D描述
  * 空间问答
  * 推理与行动计划生成（如“我如何绕开沙发去拿水？”）
* **方法：**

  * 使用统一Transformer结构将点云编码为“视觉token”，传入LLM；
  * 用户交互区域也被编码为token，支持视觉+语言多模态输入；
  * 指令微调数据集中包含多轮交互与任务计划指令；
  * 无需多视角图像，直接基于点云运行；
  * 在多个3D任务上优于其他模型，体现3D LLM的交互理解与推理潜力。

### **Chat-3D: Data-efficiently Tuning LLM for Universal Dialogue of 3D Scenes**

* **输入：** 室内3D场景（点云或mesh）+ 自然语言对话
* **输出任务：** 自由对话式的场景问答与解释，如“这是什么？它在房间哪个方向？”
* **方法：**

  * 将预训练3D视觉编码器的特征与语言嵌入空间对齐；
  * 三阶段微调策略：

    1. 粗对齐3D-语言空间（用合成或伪数据）；
    2. 使用真实3D数据集（如ScanQA）微调；
    3. 使用高质量3D指令数据集进行Instruction tuning；
  * 引入“对象中心式提示机制”（object-centric prompting）使对话聚焦于具体物体；
  * 用极少数据达成高质量性能，在某些任务上达成GPT-4的75%以上表现。

### **Visual Programming for Zero-shot Open-Vocabulary 3D Visual Grounding**

* **输入：** 室内3D场景 + 任意语言查询（如“靠窗那个坐着用的东西”）
* **输出任务：** 零样本3D指代解析（返回对应的3D区域/框），同时支持解释说明
* **方法：**

  * 使用大语言模型（如ChatGPT）进行“程序推理”，将问题拆解为子任务；
  * 组装多个可调用模块，如：

    * 全局分析模块（获取颜色、位置等）
    * 视角依赖模块（判断空间上下/前后）
    * 功能/逻辑模块（如“能坐的物体”）
  * 模块之间通过语言模型引导构建“视觉程序”，实现多轮推理；
  * 模型无需任何3D训练数据，仅靠模块+语言模型推理即可，在多个任务上超过有监督模型。

### **3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer**

* **输入：** 点云场景（含多个物体），支持用户点击或语言选定区域
* **输出任务：**

  * 3D问答与对话
  * 指代解析（文本描述 → 物体分割）
  * 场景理解与指令回应
* **方法：**

  * 提出“Omni Superpoint Transformer”（OST），集三功能于一体：

    1. 超点选择器（从点云中聚合重要区域）
    2. 视觉交互编码器（用户选中区域转为token）
    3. 分割解码器（可返回指代对象的mask）
  * 使用混合训练目标（感知+语言对齐）进行预训练；
  * 无需多阶段、多视角流程，统一直接端到端处理；
  * 在多个任务上达成领先性能，是功能全面的通用型3D语言模型。

### **ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities**

* **输入：** 室内3D场景（如ScanNet）+ 带有推理需求的问题（如“哪个能切东西的物体在桌子上？”）
* **输出任务：** 推理-结合-定位（即文字推理 + 输出该对象的3D位置）
* **方法：**

  * 构建ScanReason数据集，含10K问答对，涵盖五类推理：逻辑、比较、功能、属性等；
  * 提出ReGround3D框架：

    * 第一阶段：使用多模态LLM进行语言-图像层级的推理；
    * 第二阶段：根据推理结果在3D场景中进行目标定位（grounding）；
  * 支持“链式推理+定位”，即边推理边在场景中验证；
  * 在复杂推理定位任务上效果显著，提出了向更智能3D问答系统发展的路线。

## 3D物体

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