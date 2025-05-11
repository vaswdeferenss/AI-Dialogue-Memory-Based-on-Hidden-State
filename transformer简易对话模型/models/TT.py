import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import jieba
import logging

from text_processor import read_vectors
from transformer简易对话模型.model_components import Encoder,Decoder,memory
from transformer简易对话模型.my_dataset import txt

"""
【transformer编码器 - transformer解码器】架构的模型
简称 TT
"""

class TT(nn.Module):
    def __init__(self,device,max_length=20,topn=100000,
                 dict_path='C:\\machine_learning\\text_processor\\word_vector.iter5'):
        super().__init__()
        self.device = device
        self.max_length = max_length    #序列长度
        self.topn=topn

        embedding_matrix, word_to_index, _, embedding_dim = read_vectors(dict_path, topn)       # 加载词典
        print('词典加载完')

        self.word_to_index = word_to_index
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix.to(device), freeze=False)# 词向量层

        self.encoder=Encoder(device)        #编码层
        self.decoder=Decoder(device,topn)   #解码层

        self.to(device)

    #训练前向传播
    def forward(self,src,tgt,src_padding_mask=None,tgt_padding_mask=None,new_round=False,batch_size=16):
        """
        输入为     [batch_s,len]
        输出为     [batch_s,len,vocab_s]
        """
        src=self.embedding(src)
        tgt=self.embedding(tgt)

        y=self.encoder(src,src_padding_mask)
        y=self.decoder(y,tgt,src_padding_mask,tgt_padding_mask)
        return y

    #训练
    def model_train(self, data_path,data_len=2000000,batch_size=16, epochs=20, lr=None,params_p=None):
        if lr is None:
            lr = [2e-4, 1e-4, 2e-4]

        if params_p is not None:
            try:
                checkpoint = torch.load(params_p, map_location=self.device)
                self.encoder.load_state_dict(checkpoint['encoder'])
                self.decoder.load_state_dict(checkpoint['decoder'])
                # 加载embedding权重（保持冻结状态与初始化一致）
                self.embedding.weight.data.copy_(checkpoint['embedding'])
                print(f"成功从 {params_p} 加载预训练参数")
            except Exception as e:
                print(f"参数加载失败: {str(e)}")

        dataset=txt(data_path,self.word_to_index,data_len)      #数据
        dataloader= DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)

        loss_fun=nn.CrossEntropyLoss(ignore_index=0)            #损失函数与优化器
        optimizer_encoder=optim.AdamW(self.encoder.parameters(), lr=lr[0], betas=(0.9, 0.999), weight_decay=1e-5)
        optimizer_decoder=optim.AdamW(self.decoder.parameters(), lr=lr[2], betas=(0.9, 0.999), weight_decay=1e-5)

        best_loss = 99999999.0
        for epoch in range(epochs):
            self.train()
            loss_mean=0

            for x,y in dataloader:      #形状为  [batch_size,num_turns,len]*2
                x = x.to(self.device)
                y = y.to(self.device)

                for src,tgt in zip(x.unbind(dim=1),y.unbind(dim=1)):    #形状为  [batch_size,len]*2
                    src_padding_mask = (src == 0).to(self.device)       #填充掩码
                    tgt_padding_mask = (tgt == 0).to(self.device)

                    src_padding_mask[:, 0] = False                      #避免全为Ture导致Nan
                    tgt_padding_mask[:, 0] = False

                    optimizer_encoder.zero_grad()                       #归零
                    optimizer_decoder.zero_grad()

                    output=self.forward(src,tgt,src_padding_mask,tgt_padding_mask,batch_size)#形状为  [batch_size,len]

                    output = output.reshape(-1, output.size(-1))
                    tgt = tgt.reshape(-1)                           #扁平化
                    print('@', output,output.shape,'\n',tgt,tgt.shape,'\n')

                    """判断Nan"""
                    valid_targets = (tgt != 0).any()
                    if not valid_targets:
                        continue
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print("模型输出中存在 NaN/Inf!")
                        break
                    if torch.isnan(tgt).any() or torch.isinf(tgt).any():
                        print("目标数据中存在 NaN/Inf!")
                        break

                    loss = loss_fun(output, tgt)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                    loss.backward()

                    optimizer_encoder.step()
                    optimizer_decoder.step()
                    loss_mean += loss.item()

            print(f'epoch {epoch} loss_mean={loss_mean/data_len}')

            # if loss_mean/data_len < best_loss:
            #     self.save_core_params(params_p)  # 只保留最优参数
            #     best_loss = loss_mean/data_len
            #     print('保存成功')

    # 工作
    def work(self, params_p):

        try:
            checkpoint = torch.load(params_p, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.embedding.weight.data.copy_(checkpoint['embedding'])
            print(f"模型 {params_p} 开始工作，输入‘@exit’结束对话")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return

        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

        jieba.setLogLevel(logging.WARNING)
        while True:
            user_input = input('用户输入: ')
            if user_input == '@exit':
                print('对话已结束，感谢您的使用。')
                return

            # 将输入转为编号矩阵
            seg_list = jieba.lcut(user_input)
            src_indices = [0] * self.max_length
            for i, word in enumerate(seg_list[:self.max_length]):
                src_indices[i] = self.word_to_index.get(word, 0)
            src = torch.tensor(src_indices, device=self.device).unsqueeze(0)

            # 生成初始目标矩阵
            tgt_indices = torch.zeros((1, self.max_length), dtype=torch.long, device=self.device)
            tgt_indices[0, 0] = self.word_to_index['@']

            # 编码阶段
            src_embed = self.embedding(src)
            src_pad_mask = (src == 0).to(self.device)
            memory = self.encoder(src_embed, src_pad_mask)

            for step in range(1, self.max_length):

                # 处理目标序列
                tgt_embed = self.embedding(tgt_indices)
                tgt_pad_mask = (tgt_indices == 0)

                # 调用解码器
                output = self.decoder(tgt_embed, memory, tgt_pad_mask)

                # 连接目标序列
                next_logits = output[0, step - 1, :]
                next_token = torch.argmax(next_logits)
                tgt_indices[0, step] = next_token

                if next_token.item() == 0:
                    break

            response = []
            for idx in tgt_indices[0].tolist()[1:]:  # 跳过起始符@
                if idx == 0:
                    break
                response.append(self.index_to_word.get(idx, '<UNK>'))

            print('模型回复:', ''.join(response))

    #参数保存
    def save_core_params(self, path):
        """仅保存模型主干参数，训练状态"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'word_to_index': self.word_to_index,
            'embedding': self.embedding.weight.data
        }, path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datapath='C:\\machine_learning\\transformer简易对话模型\\data\\LCCC-base_train.json'

    model=TT(device)
    model.model_train(datapath,batch_size=16,data_len=256,params_p='parameters\\TLT.pth')