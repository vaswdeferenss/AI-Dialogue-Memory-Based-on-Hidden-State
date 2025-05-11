import torch
import torch.nn as nn

class MemoryBlock(nn.Module):
    """基础记忆模块，包含多层LSTM和对应的归一化层"""

    def __init__(self, input_dim, layer_dims, dropout=0.05):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        prev_dim = input_dim

        # 构建LSTM层和对应的归一化层
        for dim in layer_dims:
            self.layers.append(
                nn.LSTM(prev_dim, dim, num_layers=2,
                        batch_first=True, dropout=dropout)
            )
            self.layer_norms.append(nn.LayerNorm(dim))
            prev_dim = dim

        # 初始化状态容器
        self.states = [None] * len(layer_dims)

    def reset_states(self, batch_size, device):
        """重置隐藏状态"""
        for i, layer in enumerate(self.layers):
            hidden_dim = layer.hidden_size
            h = torch.randn(2, batch_size, hidden_dim).to(device)
            self.states[i] = (h, h.clone())

    def forward(self, x, new_round=False, batch_size=None):
        """前向传播"""
        if new_round and batch_size:
            self.reset_states(batch_size, x.device)

        current_input = x
        for i, (lstm, ln) in enumerate(zip(self.layers, self.layer_norms)):
            h, c = self.states[i] if self.states[i] is not None else (None, None)
            out, (new_h, new_c) = lstm(current_input, (h, c) if h is not None else None)
            self.states[i] = (new_h.detach(), new_c.detach())
            current_input = ln(out)
        return current_input


class Memory(nn.Module):
    """整合的记忆模块"""

    def __init__(self, device, dim=300, dropout=0.05):
        super().__init__()
        self.device = device
        self.dim = dim

        # 创建两个记忆模块
        self.emotional_block = MemoryBlock(
            input_dim=dim,
            layer_dims=[dim * 2, dim],
            dropout=dropout
        )
        self.information_block = MemoryBlock(
            input_dim=dim,
            layer_dims=[dim * 2, dim],
            dropout=dropout
        )

        # 可学习权重参数
        self.w_emo = nn.Parameter(torch.randn(1))
        self.w_inf = nn.Parameter(torch.randn(1))

        self.to(device)

    def forward(self, src, new_round=False, batch_size=16):
        src = src.to(self.device)

        # 前向传播并融合结果
        emo_output = self.emotional_block(
            src, new_round=new_round, batch_size=batch_size
        )
        inf_output = self.information_block(
            src, new_round=new_round, batch_size=batch_size
        )

        # 加权融合
        src = (src + self.w_inf * inf_output) * self.w_emo * emo_output
        return src

if __name__ == "__main__":
    test = torch.randn(16, 20, 300)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Memory(device)
    y = model(test, new_round=True, batch_size=16)
    print(y.shape)  # 应保持输出形状 [16, 20, 300]