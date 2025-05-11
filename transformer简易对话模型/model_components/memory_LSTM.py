import torch
import torch.nn as nn

"""
类LSTM记忆装置
输入输出都为     [batch_size,len,dim=300]
"""

class memory(nn.Module):
    def __init__(self,device,dim=300,dropout=0.05):
        super().__init__()
        self.to(device)
        self.device=device
        self.dim=dim

        self.emotional_memory = nn.ModuleList([     #情绪记忆
            nn.LSTM(self.dim, self.dim*2, 2, batch_first=True,dropout=dropout),
            nn.LSTM(self.dim*2, self.dim, 2, batch_first=True,dropout=dropout)
        ])

        self.information_memory = nn.ModuleList([   #信息记忆
            nn.LSTM(self.dim, self.dim*2, 2, batch_first=True,dropout=dropout),
            nn.LSTM(self.dim*2, self.dim, 2, batch_first=True,dropout=dropout)
        ])

        self.emotional_ln=nn.ModuleList([       #情绪记忆归一化层
            nn.LayerNorm(self.dim*2),
            nn.LayerNorm(self.dim),
        ])

        self.information_ln=nn.ModuleList([     #信息记忆归一化层
            nn.LayerNorm(self.dim*2),
            nn.LayerNorm(self.dim),
        ])

        self.dropout=nn.Dropout(dropout)

        #情绪与信息权重
        self.w_emo = nn.Parameter(torch.randn(1))
        self.w_inf = nn.Parameter(torch.randn(1))

        #状态
        self.state_emo1_h = None
        self.state_emo1_c = None
        self.state_emo2_h = None
        self.state_emo2_c = None
        self.state_inf1_h = None
        self.state_inf1_c = None
        self.state_inf2_h = None
        self.state_inf2_c = None

        self.to(device)

    def forward(self, src, new_round=False, batch_size=16):
        src = src.to(self.device)

        # 初始化所有LSTM状态 (关键修复点)
        if new_round:
            # 初始化第一层LSTM状态
            self.state_emo1_h = torch.randn(2, batch_size, self.dim * 2, device=self.device).detach()
            self.state_emo1_c = torch.randn(2, batch_size, self.dim * 2, device=self.device).detach()
            self.state_inf1_h = torch.randn(2, batch_size, self.dim * 2, device=self.device).detach()
            self.state_inf1_c = torch.randn(2, batch_size, self.dim * 2, device=self.device).detach()

            # 初始化第二层LSTM状态
            self.state_emo2_h = torch.randn(2, batch_size, self.dim, device=self.device).detach()
            self.state_emo2_c = torch.randn(2, batch_size, self.dim, device=self.device).detach()
            self.state_inf2_h = torch.randn(2, batch_size, self.dim, device=self.device).detach()
            self.state_inf2_c = torch.randn(2, batch_size, self.dim, device=self.device).detach()

        # 处理情绪记忆第一层LSTM
        emo_output, (h_emo1, c_emo1) = self.emotional_memory[0](
            src, (self.state_emo1_h, self.state_emo1_c)
        )
        self.state_emo1_h, self.state_emo1_c = h_emo1.detach().clone(), c_emo1.detach().clone()
        emo_output = self.emotional_ln[0](emo_output)

        # 处理信息记忆第一层LSTM
        inf_output, (h_inf1, c_inf1) = self.information_memory[0](
            src, (self.state_inf1_h, self.state_inf1_c)
        )
        self.state_inf1_h, self.state_inf1_c = h_inf1.detach().clone(), c_inf1.detach().clone()
        inf_output = self.information_ln[0](inf_output)

        # 处理情绪记忆第二层LSTM
        emo_output, (h_emo2, c_emo2) = self.emotional_memory[1](
            emo_output, (self.state_emo2_h, self.state_emo2_c)
        )
        self.state_emo2_h, self.state_emo2_c = h_emo2.detach().clone(), c_emo2.detach().clone()
        emo_output = self.emotional_ln[1](emo_output)

        # 处理信息记忆第二层LSTM
        inf_output, (h_inf2, c_inf2) = self.information_memory[1](
            inf_output, (self.state_inf2_h, self.state_inf2_c)
        )
        self.state_inf2_h, self.state_inf2_c = h_inf2.detach().clone(), c_inf2.detach().clone()
        inf_output = self.information_ln[1](inf_output)

        src = (src + self.w_inf * inf_output) * self.w_emo * emo_output+src
        src=self.dropout(src)
        return src

if __name__ == "__main__":
    test=torch.randn(32,20,300)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=memory(device)
    y=model(test,True,32)
    print(y)
