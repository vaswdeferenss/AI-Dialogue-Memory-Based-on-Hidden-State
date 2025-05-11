import torch
import torch.nn as nn

"""
编码器
输入为     [batch_size,len,dim=300]
输出为     [batch_size,len,dim=300]
"""

#位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=300, max_len=20):
        super().__init__()

        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.encoding,mean=0,std=0.02)

    def forward(self, x):
        """输入输出应为[batch_size,len,dim]"""
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].unsqueeze(0)  #确保能正确相加

#编码层
class Encoder_Layer(nn.Module):
    def __init__(self,d_model=300,num_heads=10,dim_feedforward=2048,dropout=0.1):
        super().__init__()

        self.attn=nn.MultiheadAttention(d_model,num_heads,dropout=dropout,batch_first=True)#自注意力层

        self.ffn=nn.Sequential(#前馈神经网络
            nn.Linear(d_model,dim_feedforward),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward,d_model)
        )

        self.norm1=nn.LayerNorm(d_model)#自注意力归一化层
        self.norm2=nn.LayerNorm(d_model)#前馈网络归一化层
        self.dropout1=nn.Dropout(dropout)#自注意力裁剪层
        self.dropout2=nn.Dropout(dropout)#前馈网络裁剪层

    def forward(self,src,  src_key_padding_mask=None):
        """输入输出应为[batch_size,len,dim]"""

        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.bool()

        output = self.norm1(src)
        output, _ = self.attn(  # 注意力
            src, src, src,
            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(output)

        output = self.norm2(src)
        output = self.ffn(output)
        src = src + self.dropout2(output)

        return src

#编码器
class Encoder(nn.Module):
    def __init__(self, device,embedding_dim=300,dropout=0.1,max_len=20, num_head=6, num_layers=6, save_path='my_params//lstm_model_params.pth'):
        super().__init__()
        self.to(device)

        self.device=device

        self.positional_encoding=PositionalEncoding(embedding_dim,max_len)  #位置编码
        self.dropout=nn.Dropout(dropout)

        self.layers=nn.ModuleList([ #堆叠编码层
            Encoder_Layer(embedding_dim,num_head,dropout=dropout) for _ in range(num_layers)
        ])
        self._reset_parameters()
        self.to(device)

    def _reset_parameters(self):    #初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                p.data.clamp_(-5,5)

    def forward(self, src,key_padding_mask=None):
        """输入应为[batch_size,len]"""
        """输入输出应为[batch_size,len,dim]"""

        src = src.to(self.device)

        src=self.positional_encoding(src)         # 位置编码
        src=self.dropout(src)
        for layer in self.layers:                 #注意力层
            src=layer(src,key_padding_mask)

        return src
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test=torch.randn(32,20,300)

    model=Encoder(device,300)
    print(model(test))