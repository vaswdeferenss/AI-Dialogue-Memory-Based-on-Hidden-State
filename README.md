# TLT: 基于Transformer与LSTM记忆增强的对话模型

Transformer-LSTM-Transformer的级联结构（TLT）

👋 这是一个探索性项目，旨在研究如何将LSTM记忆模块深度集成到Transformer架构中，以构建具备长期对话记忆能力的生成式对话模型。传统显式记忆机制存在信息泄露风险与效率问题，本项目尝试通过隐藏状态实现"记忆与模型一体化"，为对话模型提供更优雅的记忆方案。

#### 作者想说的：
由于我个人能力的原因，或者是中文模型的原因，或者是词向量模型维度只有300的原因，这个模型的对话表现能力很低，大概只能输出一些连续的有关联的词语，那根本算不上句子，尽管我已经努力仿照transformer论文去实现了。

说正事吧，为什么使用LSTM，其实我核心的目的是拥有一个状态保存的办法，至于用什么方法进行状态并没有在意，一个LSTM组件就可以保存两种状态，想想都很方便，所有就使用了。

## 项目概述

### 项目背景

在传统对话系统中，显式记忆模块常面临三个核心问题：
1. **安全隐患** - 文本记忆信息易被恶意窃取
2. **效率瓶颈** - 长记忆导致注意力复杂度呈平方增长
3. **维护困难** - 独立记忆模块增加维护复杂度

### 核心目标
- **记忆融合**：通过LSTM隐藏状态实现隐式加密记忆，避免传统显式记忆的存储与计算开销
- **架构创新**：构建Transformer-LSTM-Transformer的级联结构（TLT），探索混合架构的可能性
- **效率优化**：保持O(n)时间复杂度，相比传统记忆机制具有更好的可扩展性

### 模型架构

1. **Transformer编码器**：提取输入文本的深层语义特征
2. **LSTM记忆模块**：双通道LSTM网络分别处理情感记忆与事实记忆
3. **Transformer解码器**：融合记忆信息生成自然回复

### 技术亮点
- 动态记忆重置机制
- 双通道记忆分离处理
- 自适应记忆融合权重
- 重复惩罚生成策略

## 项目结构
```
project-root/
├── text_processor/                # 文本处理模块
│   ├── word_vector.iter5          # 预训练词向量文件（未上传）
│   └── 词向量模型的倒入.py         # 词向量加载器
├── transformer简易对话模型/        # 核心实现
│   ├── model_components/          # 模型组件
│   │   ├── decoder_transformer.py      #解码器
│   │   ├── encoder_transformer.py      #编码器
│   │   └── memory_LSTM.py              #记忆器
│   ├── models/                    # 完整模型
│   │   ├── TLT.py                 # 带记忆的完整模型
│   │   └── TT.py                  # 无记忆对照模型
│   ├── data/                      # 示例数据    （未上传）
│   ├── parameters/                # 模型参数存储（未上传）
│   ├── my_dataset.py              # 数据加载器
│   └── main.py                    # 主程序入口
```

## 使用说明

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- jieba 0.42+
- numpy 1.22+

### 快速开始
1. **训练模型**
```python
# main.py
def train_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TLT(device)
    model.model_train(
        datapath='path/to/dataset',
        batch_size=32,
        epochs=10,
        params_p='parameters/TLT.pth'
    )
```

2. **启动对话**
```python
# main.py
def work_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TLT(device)
    model.work1('parameters/TLT.pth')
```

### 参数配置
| 参数名称          | 默认值   | 说明                     |
|-------------------|---------|--------------------------|
| max_length        | 20      | 最大序列长度             |
| num_head          | 6       | 注意力头数               |
| dim_feedforward   | 1024    | 前馈网络维度             |
| repetition_penalty| 1.2     | 重复生成惩罚系数         |

## 实验验证

### 初步成果
- 在多轮对话中观察到上下文关键词有关联性
- 记忆模块梯度传递正常，参数更新有效
- 在RTX 4070上实现100ms级响应速度

### 现存问题
❗ 当前版本局限性：
- 生成语句通顺度有待提升
- 长期记忆保持能力不足
- 对复杂语义理解有限

## 改进方向
- [ ] 分部训练避免LSTM层的过拟合
- [ ] 引入预训练语言模型增强语义理解
- [ ] 实现分层记忆结构
- [ ] 添加注意力门控机制
- [ ] 优化记忆持久化策略

## 致谢
本项目受以下项目启发：
- [Transformer](https://arxiv.org/abs/1706.03762)
- [DialoGPT](https://github.com/microsoft/DialoGPT)
- [PLATO](https://arxiv.org/abs/1910.07931)

欢迎提交Issue和PR！🤝
