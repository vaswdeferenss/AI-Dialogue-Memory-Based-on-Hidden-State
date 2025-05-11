from models import TLT
import torch

# 调用该函数开始训练
def train_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datapath='D:\\mc_leaning\\transformer简易对话模型\\data\\LCCC-base_train.json'

    model=TLT(device)
    model.model_train(datapath,
                      batch_size=32,
                      data_len=1024+128*2,
                      epochs=1,
                      lr=[20e-5,1e-4 , 20e-5],
                      params_p='parameters\\TLT.pth')

# 调用该函数开始对话
def work_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params_path='parameters\\TLT.pth'

    model=TLT(device)
    model.work1(params_path)

if __name__ == "__main__":
    #选择你要进行的操作
    # train_main()
    work_main()