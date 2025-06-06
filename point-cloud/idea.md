## 静态物体点云公开数据集

为支持多角度模拟和协同识别，我们需要**包含完整单个物体点云**（而非复杂场景）的公开数据集，并尽可能涵盖多类别物体。以下是几种常用的数据集：

* **ModelNet**：Princeton发布的CAD模型数据集，涵盖**662类**物体共约127,915个3D模型。常用子集包括ModelNet10和ModelNet40，其中**ModelNet40**包含40类常见物体的3D模型，共12,311个（训练9,843，测试2,468），每个模型表面可均匀采样出点云。这些点云通常经过预处理（平移到原点并缩放至单位球），非常适合作为单个物体的基准点云数据。

* **ShapeNet**：大规模3D物体数据库，收录了**约300万+模型，4000+类别**。其中核心子集ShapeNetCore提供了**51,300个模型，分属55类**。ShapeNet的模型带有丰富注释（如分割、对齐等），可方便地转换为点云格式，用于多类别物体识别研究。

* **ScanObjectNN**：一个源自真实扫描的物体点云数据集，从室内扫描数据(SceneNN、ScanNet)提取出**15类**、共15,000个物体点云。相比合成数据，ScanObjectNN的点云更加稀疏且带有自遮挡和噪声，因此对算法具有更大挑战性。然而，它提供了真实场景下单个物体的点云，有助于研究在不完美数据下的识别效果。

* **Objaverse**：一个最新的大规模3D对象库，包含来自互联网的多种物体模型。PointLLM论文构建了Objaverse点云数据，用于指令学习，**包含约66万**个彩色物体点云。每个物体点云保存为numpy文件(npy)，大小为(8192,6)，即**8192个点，每点含XYZ坐标和RGB颜色**（归一化到\[0,1]）。Objaverse覆盖的物体类别极其丰富，适合需要**多类别、多样化形状**的研究，并且其数据格式与PointLLM模型**直接兼容**。

上述数据集均提供**单个物体级**的完整点云，满足多传感器角度切分的模拟需求。其中ModelNet和ShapeNet为合成数据，点云均衡且完整；ScanObjectNN和Objaverse则提供真实感更强或规模更大的物体点云。通过将这些数据转换成所需格式（例如采样一定数量点并添加颜色），即可用于后续PointLLM模型输入。

## 点云多角度切分方法

在模拟多个LiDAR从不同角度观测同一静态物体时，我们需要将完整物体点云“切分”成不同视角的子点云。每个视角对应某个传感器位置，只包含该视角**正面可见**的点。常用的方法有以下两种：

**1. 基于角度扇区的切分：**
将360°视野按视角数量划分为若干扇区，每个扇区对应一个LiDAR视角。具体步骤为：

* 以物体质心为原点，计算每个点在水平面的**方位角**θ，例如 `θ = arctan2(y, x)`（以度为单位0°\~360°）。
* 设定视角数量*N*（默认3个），则每个视角扇区覆盖约`360°/N`的范围。例如3视角情况下，可将水平面分成3个120°扇区。
* 根据点的方位角将点云划分：落入第1扇区的点属于视角1，落入第2扇区的属于视角2，依此类推。必要时还可结合垂直角度限制视场高度范围。简单示例代码如下：

```python
import numpy as np
points = np.load('object.npy')  # (N,3)或(N,6)数组，包含XYZ(+RGB)
angles = np.degrees(np.arctan2(points[:,1], points[:,0])) % 360  # 计算水平角度
N = 3  # 视角数量
sector_size = 360.0 / N
subclouds = []
for i in range(N):
    start = i * sector_size
    end = (i+1) * sector_size
    mask = (angles >= start) & (angles < end)
    subclouds.append(points[mask])
```

这样即可得到三个子点云`subclouds[0..2]`，分别对应不同方向的大致**120°视场**。每个子点云近似模拟了一个LiDAR从相应方向看到的物体点云。需要注意，此方法假设物体大致均匀，未考虑点的遮挡关系；但在物体相对稠密均匀时，简单按角度划分已能代表不同视角的主要信息。

**2. 基于虚拟相机/传感器模拟：**
更精确的方法是使用3D引擎或点云库模拟LiDAR采样过程。例如，如果有物体的三角网格模型，可以放置虚拟相机在指定角度，对模型进行**深度渲染**获取可见点云。这实际上会自动考虑遮挡和视场限制。

使用Open3D库可以方便地实现这一点：将物体加载为TriangleMesh，设置视角后调用`capture_depth_point_cloud()`直接生成该视角下的点云。Open3D会对背面自动进行裁剪，相当于**实现了视角的背面剔除和深度截断**。例如：

```python
vis = open3d.visualization.Visualizer()
vis.add_geometry(triangle_mesh)         # 加载物体网格
# ... 设置视角参数，例如旋转角度 ...
vis.capture_depth_point_cloud("view1.pcd", convert_to_world_coordinate=True)
```

上述代码将直接保存当前视角下的点云到文件。通过改变相机角度重复这一过程，可以得到多个视角的点云片段。相比按角度阈值硬切，这种方法更接近真实LiDAR采样（会产生稀疏不均的点云，并准确反映可见表面）。不过需要物体的网格模型作为输入，以及一定的渲染开销。

**小结：**针对静态物体点云，简单的角度扇区切分方法实现容易且可自定义视角数量和范围；而借助Open3D等工具进行模拟则更精细准确。实践中可根据需要选择：例如在快速验证阶段使用角度切分，在逼真仿真阶段采用虚拟LiDAR扫描。需要注意的是，多视角点云通常是**有局部缺失（partial）**的，这与真实传感器获取结果一致，也符合文献中对**单视角部分点云**分类精度下降的观察。

## PointLLM的使用说明

PointLLM是一个将点云理解能力融入LLM的大模型，它能够对输入的**彩色点云**进行物体识别和描述。本节说明PointLLM的使用方法，包括数据准备、模型安装与推理流程。

### 数据格式要求

PointLLM模型要求输入点云为**固定尺寸的点集**，默认使用每个物体8192个点，且每个点含有6维特征。具体而言，每个点表示为`(x, y, z, r, g, b)`，其中坐标xyz通常规范化在一定尺度内，颜色rgb为0\~1归一化值。如果数据集原始格式不同，需要进行转换：

* **点数**：如果物体点云点数不是8192，可通过随机采样或FPS(Farthest Point Sampling)重采样到统一数量（PointLLM提供了ModelNet40测试集8192点采样文件）。模型在训练时采用8192点，因此推理时也保持一致。
* **颜色**：对于没有颜色的点云，可统一设定颜色值为0（黑）或使用法线/高度等替代颜色信道。PointLLM是针对彩色点云设计的，如无颜色信息也能运行，但可能无法发挥对外观的理解优势。

训练用的数据例如Objaverse点云已经处理为.npy文件，每个物体对应一个数组文件，形状(8192,6)。在使用自己的数据时，建议仿照这一格式组织，并确保归一化坐标和适当的颜色值范围。

### 安装与环境准备

PointLLM提供了开源代码和模型权重，可在Linux环境下部署。根据官方说明，测试环境包括Ubuntu 20.04、Python 3.10、PyTorch 2.0.1、CUDA 11.7，以及相应NVIDIA驱动等。安装步骤如下：

1. **获取代码**：从GitHub克隆PointLLM仓库并进入目录：

   ```bash
   git clone https://github.com/OpenRobotLab/PointLLM.git
   cd PointLLM
   ```

2. **创建环境**：建议使用Conda创建独立环境：

   ```bash
   conda create -n pointllm python=3.10 -y
   conda activate pointllm
   ```

3. **安装依赖**：通过pip安装项目依赖：

   ```bash
   pip install --upgrade pip  # 确保pip最新以支持PEP 660
   pip install -e .
   ```

   如需训练模型，还需安装`ninja`、`flash-attn`等额外包。上述步骤完成后，代码环境就准备就绪。

4. **准备数据**：下载模型作者提供的训练/评估数据。例如Objaverse点云数据（约77GB）可以从HuggingFace获取；若进行ModelNet40评估，需要下载官方采样的测试点云文件`modelnet40_test_8192pts_fps.dat`放置于`PointLLM/data/modelnet40_data`目录。保证数据路径与代码中默认配置一致，或在运行时通过参数指定。

5. **获取模型权重**：PointLLM提供了预训练模型（如7B和13B参数量版本）托管在HuggingFace仓库上。运行推理脚本时，会自动从`RunsenXu/PointLLM_7B_v1.2`等地址下载权重。确保运行环境能访问互联网以下载模型（或手动下载放置到指定目录）。

### 推理流程

PointLLM支持**开放词汇物体分类**和\*\*点云描述（captioning）\*\*等任务。使用预训练模型进行推理的流程一般如下：

1. **选择任务与提示词**：对于分类，可选择开放集合分类（输出物体名称）或闭集零样本分类（针对特定已知类别集）。对于描述任务，模型将生成自然语言描述。PointLLM使用*prompt*来引导输出，在命令行参数中用`--task_type`和`--prompt_index`指定。例如：

   * 开放词汇分类（Objaverse数据集）：

     ```bash
     python pointllm/eval/eval_objaverse.py --model_name RunsenXu/PointLLM_7B_v1.2 \
            --task_type classification --prompt_index 0 
     ```
   * 物体描述（Objaverse数据集）：

     ```bash
     python pointllm/eval/eval_objaverse.py --model_name RunsenXu/PointLLM_7B_v1.2 \
            --task_type captioning --prompt_index 2 
     ```
   * 闭集零样本分类（ModelNet40数据集）：

     ```bash
     python pointllm/eval/eval_modelnet_cls.py --model_name RunsenXu/PointLLM_7B_v1.2 \
            --prompt_index 1 
     ```

   以上命令会加载指定模型权重，并对默认数据集逐条推理，将结果保存在`{model_name}/evaluation/`目录下。`prompt_index`可以选0或1来使用不同表述的提示词模板（例如“这是什么物体？”或“请识别该对象。”）。

2. **输入数据**：推理脚本会自动从`data/`目录读取对应任务的数据文件。例如Objaverse分类脚本默认读取`data/objaverse_data`中的点云文件和注释。若要测试自定义点云，可将点云文件路径传入或修改脚本读取逻辑。确保输入点云已满足前述格式要求（numpy数组或pt文件，含正确维度）。

3. **运行推理**：执行上述脚本后，模型会对每个输入点云生成结果。例如分类任务输出每个对象的预测类别（文字描述）和置信度；描述任务输出一段文字描述。PointLLM采用**生成式**方式回答，因此分类结果常以自然语言形式呈现（例如：“This is a chair”）。结果会以JSON列表形式保存，包含对象ID、模型输出文本、（如适用）真值标签等信息。

4. **查看结果**：可以编写解析脚本读取保存的结果文件，从中提取模型识别的类别或者描述文本进行分析。如果需要评价性能，PointLLM提供了**GPT-4/ChatGPT评估**和传统指标评估脚本，可参考其README的说明执行。

此外，PointLLM还提供了**交互式对话接口**。通过运行Gradio演示程序，可在浏览器界面上传点云并与模型交互。启动命令示例：

```bash
PYTHONPATH=$PWD python pointllm/eval/chat_gradio.py --model_name RunsenXu/PointLLM_7B_v1.2 \
       --data_path data/objaverse_data
```

启动后可以可视化点云并让模型回答关于该物体的问题，实现更灵活的使用方式。

*（注意：运行如此体量的大模型需要较高的计算资源。7B参数模型运行推理约需≥16GB GPU显存（FP16精度），13B模型需求更高。使用前请确认硬件条件。）*

### 总结

PointLLM的使用流程可以概括为：准备符合格式的数据、安装环境和模型，选择任务脚本并运行得到结果。其训练好的模型在3D物体**开放集识别**方面表现突出，可以产生对点云的合理分类结果或自然语言描述，有力证明了将LLM与点云特征融合的效果。

## 多视角识别结果融合策略

当来自多个视角的LiDAR分别对同一物体进行识别后，我们需要融合这些视角的结果，以给出更准确可靠的判决。融合策略可以在**决策层**结合不同传感器/模型的输出。以下介绍几种常见且有效的融合方法：

### 基于Dempster-Shafer证据理论的融合

Dempster-Shafer (D-S)证据理论是一种处理不确定性信息融合的框架。它将各传感器的判别结果视为对不同类别的**信度分配**(Belief mass)。具体做法是：为每个分类器输出一个**基本概率分配(BPA)**，即对每一类别和“未知”集合赋予一个信任度(mass)。然后使用**Dempster合规则**逐对组合这些证据，从而得到综合的信任分布。D-S融合的特点在于：

* 能显式表示**不确定和冲突**：如果某视角无法确定类别，可将较大质量赋给“未知”集合；多个视角的信息冲突时，Dempster规则会通过归一化减弱冲突影响。这比简单概率乘积更健壮。
* **融合过程可解释**：通过计算**信任度Belief**和**可能性Plausibility**，可以解释不同证据如何支持某一类别以及不确定性范围。
* 实际应用：将每个LiDAR视角分类器输出的类别置信度转为BPA（例如直接以概率作为信度分配），再迭代应用Dempster合成公式，获得融合后各类别的综合信度。最后选择信度最高的类别作为识别结果。这种方法特别适用于**多传感器在信息不完备或有冲突**时的决策融合，被广泛应用于安全关键系统的可靠识别。

需要注意D-S理论在证据高度冲突时可能出现反直觉结果（如归一化因子过小导致分配偏差），对此有改进方法（如调整合成规则等）。总体而言，D-S融合提供了一套系统的方法来整合多视角分类结果，在确保准确率的同时定量表达不确定性。

### 投票融合

**投票法**是一种简单直观的决策级融合策略，常用于集成多个分类器的结果。其基本思想是让每个视角的识别结果进行“投票”，最终选择获得多数投票的类别作为融合输出。具体而言：

* **简单投票（多数表决）**：每个LiDAR视角提供一个判定类别（或Top-1候选），所有视角的投票一人一票，最后票数最多的类别获胜。例如有3个传感器，其中2个识别为“汽车”，1个识别为“卡车”，则融合结果取“汽车”。
* **平票与处理**：若出现平票情况，可以引入预定规则如：参考平均置信度较高者、引入少量随机决策，或进一步求助其他融合方法。
* **优缺点**：投票法**实现简单，易于理解**。当各视角分类器性能相近且独立时，投票能有效**降低误判概率**（多数视角纠正少数错误）。然而，如果某些视角显著优于其他，简单投票会**忽视差异**，可能降低总体性能。另外投票未考虑置信度大小，只看**是否**投某类别，这在一些场景下会损失信息。

投票融合适合快速融合多个模型意见，尤其在每个视角结果都是明确类别且可靠性相近时。如需改进，可引入**加权投票**策略。

### 置信度加权融合

置信度加权融合是在投票法基础上考虑各分类结果的**置信度或可靠性**差异，对不同视角赋予不同权重。核心思想是“让**更有把握**的观点起更大作用”，避免简单多数决策可能的误导。实现方式包括：

* **加权投票**：每个视角的票按照预先设定的权重进行计数，然后汇总权重得到各类别总分，选择最高分类别。权重可以根据各视角模型的历史准确率、传感器质量等设定。例如精度高的传感器赋权0.5，其余赋0.25，则它的票相当于其他两者之和。实践表明，如果权重合理分配，可提高整体性能。
* **概率平均（软投票）**：让每个模型输出**对各类别的概率分布**，然后对概率按权重取平均（或加权平均）。最终取平均后概率最大的类别。这等价于在“信心高”方向倾斜，比硬投票利用了更多信息。当所有分类器可信度类似时，等权重平均（即软投票）往往优于硬投票，在许多比赛中被视为集成首选策略。
* **动态权重融合**：权重不固定，根据每个输入实例的情况动态调整。例如如果某视角当前输出概率分布熵很低（很自信），则提高其权重；反之降低。这需要设计一定的函数或规则，甚至训练一个次级模型来预测各视角可信度。

置信度加权融合相对普通投票更**灵活**，能**反映传感器可靠度**差异。其效果取决于权重设定是否合理，可通过验证集调优。如果无法确定权重，也可采用等权的软投票作为起点，因为它已利用了置信度信息（概率值大小）。

### 其他融合方法

除上述方法外，还有一些可用于多视角结果融合的策略：

* **贝叶斯融合**：将各视角输出视为独立证据，根据贝叶斯公式逐类相乘概率获得后验概率，再归一化得到融合结果。这相当于**概率乘积规则**（Product Rule）。当各视角错误相互独立且精度较高时，乘积规则可强化一致的判断。然而如果某一视角对真实类别赋很低概率，乘积可能过度惩罚导致漏检。因此通常需要对概率进行平滑或设置信任下限，以防止某个视角的置信错误主导结果。

* **最大可信度原则**：直接选择所有视角中**置信度最高**的一个判断作为结果。比如一个LiDAR从正面看得很清楚，输出“自行车”概率0.95，而侧面视角只有0.6置信度，则相信正面视角的结论。这种策略简单有效，尤其当某单视角出现压倒性信心时。但单凭最高置信度可能忽略了其他视角的佐证，风险在于有时模型的过度自信未必正确。因此通常结合其他方法或对置信度进行校准后再采用max规则。

* **学习融合（Stacking）**：使用一个**次级学习器**（如逻辑回归、浅层神经网络）以各视角的输出（概率向量或类别预测）作为输入，再训练得到最终决策。该方法在充足数据下效果往往最佳，因为它可以学到不同视角错误的模式和最优组合方式。例如可训练一个模型根据三个视角对各类别的打分来预测真实类别。这种融合属于**可学习的后融合**。缺点是需要额外的训练数据和模型，在多传感器场景下部署复杂度较高，同时解释性相对较差（但可以通过查看次级模型权重获得一定解释）。

* **其他规则**：例如**投票 + 门限**（当最高票数占比不到一定阈值则判定为不确定）、**平均置信度**（对每类取各视角概率均值最高者）、**Borda计数**（将排序转换为分值融合）等。在特定应用中，还可以融入业务规则或基于类别的重要性调节融合策略。

综上，多视角融合策略应根据传感器特点和任务需求选择：如果追求**稳健性和不确定度刻画**，Dempster-Shafer理论提供了严谨框架；若要求**简单易实现**，投票或软投票是不錯的选择；当不同视角质量差异明显，可采用**加权或学习式**融合来提升性能。在实现过程中，也可以将多种方法结合，例如先用D-S融合获取综合信任度，再用简单投票作为备选校验。这些融合策略的合理应用将有效提升多LiDAR协同识别静态物体的准确性和可靠性。

**参考文献：**

1. Wu, Z. *et al.* (2015). **3D ShapeNets: A Deep Representation for Volumetric Shapes** – *ModelNet 数据集描述*
2. Chang, A. *et al.* (2015). **ShapeNet: An Information-Rich 3D Model Repository** – *ShapeNetCore 数据集描述*
3. Uy, M. *et al.* (2019). **Revisiting Point Cloud Classification: A New Benchmark Dataset** – *ScanObjectNN 数据集介绍*
4. Xu, R. *et al.* (2024). **PointLLM: Empowering Large Language Models to Understand Point Clouds** – *PointLLM代码与数据说明*
5. Andrei, M. (2022). StackOverflow Q\&A: *Generate a partial view of a mesh as a point cloud* – *Open3D获取局部视角点云的方法*
6. GeeksforGeeks – *Dempster-Shafer理论概述*
7. 王雨伯 (2023). 知识分享: *集成学习常用组合策略* – *投票法与加权投票说明*
8. Han, Y. *et al.* (2022). **Trusted Multi-View Classification with Dynamic Evidential Fusion** – *多视角后融合方法综述*
