import torch
from text_processor import read_vectors
from torch.utils.data import Dataset
import ijson

"""
数据处理器：
数据总体形状   [6000000,5,20]
索引数据形状   [5,20]*2
训练数据形状   [batch_size,5,20]*2
"""

class txt(Dataset):
    def __init__(self, p, word_to_index, data_len=6000000, max_turns=5, max_length=20):
        self.word_to_index = word_to_index
        self.max_turns = max_turns
        self.max_length = max_length
        self.data_len = data_len

        pad_sentence = [0] * self.max_length    #生成模板
        pad_turn = [pad_sentence.copy() for _ in range(self.max_turns)]
        self.data_input = torch.zeros((data_len, max_turns, max_length), dtype=torch.long)
        self.data_target = torch.zeros((data_len, max_turns, max_length), dtype=torch.long)

        processed_count = 0

        with open(p, 'r', encoding='utf-8') as d:
            convs = ijson.items(d, 'item')
            for idx, conv in enumerate(convs):      #遍历对话级
                if processed_count >= data_len:
                    break

                turns_input = []
                turns_target = []
                for i in range(len(conv) - 1):      #遍历回合级
                    sent_in = conv[i].strip().split()
                    sent_out = conv[i + 1].strip().split()

                    # 处理输入句子
                    sent_in = [self.word_to_index.get(word, 0) for word in sent_in][:self.max_length]
                    sent_in += [0] * (self.max_length - len(sent_in))

                    # 处理目标句子并插入
                    sent_out_indices = [self.word_to_index.get(word, 0) for word in sent_out] #+ [self.word_to_index['<end>']]
                    sent_out_padded = sent_out_indices[:self.max_length] + [0] * (self.max_length - len(sent_out_indices))

                    # 检查是否非全填充
                    # if not all(x == 0 for x in sent_out_padded):
                    #     start_index = self.word_to_index['<start>']
                    #     new_sent = [start_index] + sent_out_padded
                    #     new_sent = new_sent[:self.max_length]    # 截断保持长度
                    #     sent_out_padded = new_sent

                    turns_input.append(sent_in)
                    turns_target.append(sent_out_padded)

                if not turns_input:
                    continue

                # 填充或截断回合数
                if len(turns_input) > self.max_turns:
                    turns_input = turns_input[:self.max_turns]
                    turns_target = turns_target[:self.max_turns]
                else:
                    pad_num = self.max_turns - len(turns_input)
                    turns_input += [pad_sentence.copy() for _ in range(pad_num)]
                    turns_target += [pad_sentence.copy() for _ in range(pad_num)]

                self.data_input[processed_count] = torch.tensor(turns_input, dtype=torch.long)
                self.data_target[processed_count] = torch.tensor(turns_target, dtype=torch.long)
                processed_count += 1

                if processed_count / data_len*100%20 == 0:
                    print(f"训练数据已加载{processed_count / data_len*100}%")

            # 截断未使用的空间
            if processed_count < data_len:
                self.data_input = self.data_input[:processed_count]
                self.data_target = self.data_target[:processed_count]

        print(self.data_input.shape, '*2\n')

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        return self.data_input[idx], self.data_target[idx]

if __name__ == "__main__":
    word_embeddings_p = 'D:\\mc_leaning\\text_processor\\word_vector.iter5'
    embedding_matrix, word_to_index, _, embedding_dim = read_vectors(word_embeddings_p, 50000)
    print('词典加载完成')
    dataset = txt('D:\\mc_leaning\\transformer简易对话模型\\data\\LCCC-base_train.json',
                  word_to_index,
                  data_len=100,
                  max_length=32)
    print('*', [data for data in dataset[54]],dataset[54][0].shape)
