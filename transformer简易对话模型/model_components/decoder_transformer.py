import torch
import torch.nn as nn

"""
解码器
输入为     [batch_size,len,dim=300]*2
输出为     [batch_size,len,词典大小]
"""

#tgt位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=300, max_len=20):
        super().__init__()

        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.encoding,mean=0,std=0.02)  #正态分布

    def forward(self, x):
        """输入输出应为[batch_size,len,dim]"""
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].unsqueeze(0)  #确保能正确相加

#解码层
class DecoderLayer(nn.Module):
    def __init__(self, d_model=300, num_heads=10, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model,num_heads,dropout=dropout,batch_first=True)  #自注意力
        self.cross_attn = nn.MultiheadAttention(d_model,num_heads,dropout=dropout,batch_first=True) #交叉注意力

        self.lin1 = nn.Linear(d_model,dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward,d_model)
        self.relu = nn.LeakyReLU()

        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        self.LN3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,tgt_key_padding_mask=None, memory_key_padding_mask=None):

        attn_output=self.self_attn(tgt,tgt,tgt,
                                   attn_mask=tgt_mask,  # 因果掩码（防止看未来）
                                   key_padding_mask=tgt_key_padding_mask)[0]
        tgt=tgt+self.dropout(attn_output)
        tgt=self.LN1(tgt)

        attn_output=self.cross_attn(tgt,memory,memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt=tgt+self.dropout(attn_output)
        tgt=self.LN2(tgt)

        attn_output = self.lin2(self.relu(self.lin1(tgt)))  # 两层线性变换+ReLU
        tgt = tgt + self.dropout(attn_output)
        tgt = self.LN3(tgt)
        return tgt

class Decoder(nn.Module):
    def __init__(self,device,vocab_size, d_model=300, num_head=6, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=20):
        super().__init__()
        self.device=device

        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_len) # 位置编码
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([                   # 编码层
            DecoderLayer(d_model, num_head, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)                #转为独热编码

        self._reset_parameters()        # 参数初始化
        self.register_buffer('tgt_mask', self.generate_square_subsequent_mask(max_len))

        self.to(device)

    def generate_square_subsequent_mask(self, sz):# 生成逻辑掩码
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def _reset_parameters(self):# 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                p.data.clamp_(5, 5)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt=tgt.to(self.device)
        memory=memory.to(self.device)
        tgt_key_padding_mask = tgt_key_padding_mask.bool() if tgt_key_padding_mask is not None else None
        memory_key_padding_mask = memory_key_padding_mask.bool() if memory_key_padding_mask is not None else None

        tgt = self.pos_encoder(tgt)     #位置编码
        tgt = self.dropout(tgt)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory,
                        tgt_mask=self.tgt_mask[:tgt.size(1), :tgt.size(1)].bool(),  # 因果掩码
                        tgt_key_padding_mask=tgt_key_padding_mask,                  # tgt填充掩码
                        memory_key_padding_mask=memory_key_padding_mask)            # src填充掩码

        tgt=self.fc(tgt)

        return tgt
if __name__ == "__main__":
    test=torch.randn(16,20,300)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Decoder(device,50000)
    y=model(test,test)
    print(y.shape)
