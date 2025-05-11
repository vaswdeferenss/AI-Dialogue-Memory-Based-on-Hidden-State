import numpy as np
import torch
import torch.nn as nn

def read_vectors(path='word_vector.iter5', topn=0):
    lines_num, dim,length = 0, 0, 0#分别代表读取了多少行，词向量的维度
    vectors = {}        #字典，记录词与其对于向量
    iw = []             #通过顺序索引词
    wi = {}             #通过词索引顺序

    with open(path,encoding='utf-8',errors='ignore')as f:
        first_line=True #标记是否为第一行

        for line in f:
            # 如果是第一行，读取维度，并添加特殊符号
            if first_line:
                first_line=False
                length=int(line.strip().split()[0])
                dim=int(line.strip().split()[1])

                vectors['<pad>'] = np.zeros(shape=(dim,))
                vectors['<start>'] = np.random.randn(dim,)
                vectors['<end>'] = np.random.randn(dim,)

                iw.append('<pad>')
                iw.append('<start>')
                iw.append('<end>')

                wi['<pad>'] = 0
                wi['<start>'] = 1
                wi['<end>'] = 2

                lines_num += 3
                continue
            lines_num+=1
            tokens=line.rstrip().split()
            vectors[tokens[0]]=np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])

            if lines_num>=length:           #如果超出限度
                break
            if topn!=0 and lines_num>=topn: #如果已经导入了足够数量的词
                break

    for i,w in enumerate(iw):
        wi[w]=i

    embedding_matrix = torch.FloatTensor(lines_num, dim)  # 建立词向量矩阵

    for i, word in enumerate(iw):  # 复制
        if i < len(embedding_matrix):
            embedding_matrix[i] = torch.from_numpy(vectors[word])  # 需要由顺序索引

    return embedding_matrix,wi,iw,dim

if __name__ == "__main__":
    device=torch.device('cuda')
    v_path='word_vector.iter5'
    topn=50000

    embedding_matrix,wi,iw,_=read_vectors(v_path,topn)#调用函数
    # print(embedding_matrix.shape)
    #
    # embedding=nn.Embedding.from_pretrained(embedding_matrix,freeze=False).to(device)#矩阵本体载入
    #
    # word_indices=[wi[word] for word in ['你好','喜欢']if word in wi]#由词变为顺序
    # word_indices=torch.tensor(word_indices,dtype=torch.long).to(device)#将顺序变为词向量
    #
    # word_embeddings=embedding(word_indices)
    # print(word_embeddings)
    print(wi['<start>'])
    print(wi['<end>'])


