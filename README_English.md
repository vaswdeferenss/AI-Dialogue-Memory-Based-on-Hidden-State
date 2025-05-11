
# TLT: Transformer-LSTM-Transformer Cascade Structure for Memory-Enhanced Dialogue Models  

Transformer-LSTM-Transformer Cascade Structure (TLT)  

👋 This is an exploratory project aiming to study the deep integration of LSTM memory modules into Transformer architectures, with the goal of building generative dialogue models capable of long-term conversational memory. Traditional explicit memory mechanisms face risks of information leakage and efficiency issues. This project attempts to achieve "integration of memory and model" through hidden states, providing a more elegant memory solution for dialogue models.  


#### Author's Notes:  
Due to personal limitations, issues with Chinese models, or the fact that the word vector model only has 300 dimensions, the conversational performance of this model is very low. It can only output some continuously related words, barely forming coherent sentences. Although I tried my best to implement it following the Transformer paper.  

Let’s get to the point: Why use LSTM? My core goal was to have a method for state preservation, and I didn’t care much about the specific state implementation. An LSTM component can preserve two types of states, which seemed very convenient, so I used it. I believe that in the future, others might find many better methods for preserving model states. Ultimately, it’s my personal limitation that I couldn’t customize a better state preservation scheme.  

I’m a first-year university student from China. This is my first project release, originally intended to enhance my understanding of deep learning. I don’t think this project has significant research value.  


## Project Overview  

### Project Background  
Traditional dialogue systems with explicit memory modules face three core issues:  
1. **Security Risks** - Text memory information is vulnerable to malicious theft.  
2. **Efficiency Bottlenecks** - Long memories cause attention complexity to grow quadratically.  
3. **Maintenance Difficulties** - Independent memory modules increase maintenance complexity.  


### Core Objectives  
- **Memory Integration**: Achieve implicit encrypted memory through LSTM hidden states to avoid the storage and computational overhead of traditional explicit memory.  
- **Architectural Innovation**: Construct a Transformer-LSTM-Transformer cascade structure (TLT) to explore the possibilities of hybrid architectures.  
- **Efficiency Optimization**: Maintain O(n) time complexity, offering better scalability compared to traditional memory mechanisms.  


### Model Architecture  
1. **Transformer Encoder**: Extract deep semantic features from input text.  
2. **LSTM Memory Module**: A dual-channel LSTM network processes emotional memory and factual memory separately.  
3. **Transformer Decoder**: Fuse memory information to generate natural responses.  


### Technical Highlights  
- Dynamic memory reset mechanism.  
- Dual-channel memory separation processing.  
- Adaptive memory fusion weight.  
- Repetition penalty generation strategy.  


## Project Structure  
```  
project-root/  
├── text_processor/                # Text processing module  
│   ├── word_vector.iter5          # Pre-trained word vector file (not uploaded)  
│   └── 词向量模型的倒入.py         # Word vector loader  
├── transformer简易对话模型/        # Core implementation  
│   ├── model_components/          # Model components  
│   │   ├── decoder_transformer.py      # Decoder  
│   │   ├── encoder_transformer.py      # Encoder  
│   │   └── memory_LSTM.py              # Memory unit  
│   ├── models/                    # Complete models  
│   │   ├── TLT.py                 # Full model with memory  
│   │   └── TT.py                  # Control model without memory  
│   ├── data/                      # Example data (not uploaded)  
│   ├── parameters/                # Model parameter storage (not uploaded)  
│   ├── my_dataset.py              # Data loader  
│   └── main.py                    # Main program entry  
```  


## Usage Instructions  

### Environment Requirements  
- Python 3.8+  
- PyTorch 1.12+  
- jieba 0.42+  
- numpy 1.22+  


### Quick Start  

1. **Train the Model**  
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

2. **Start Conversation**  
```python  
# main.py  
def work_main():  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model = TLT(device)  
    model.work1('parameters/TLT.pth')  
```  


### Parameter Configuration  
| Parameter Name         | Default Value | Description                          |  
|------------------------|---------------|--------------------------------------|  
| max_length             | 20            | Maximum sequence length              |  
| num_head               | 6             | Number of attention heads            |  
| dim_feedforward        | 1024          | Feedforward network dimension        |  
| repetition_penalty     | 1.2           | Repetition generation penalty coefficient |  


## Experimental Validation  

### Initial Achievements  
- Observed contextual keyword relevance in multi-turn conversations.  
- Normal gradient propagation in memory modules with effective parameter updates.  
- 100ms-level response speed on RTX 4070.  


### Existing Issues  
❗ Limitations of the current version:  
- Generated sentence fluency needs improvement.  
- Insufficient long-term memory retention.  
- Limited understanding of complex semantics.  


## Improvement Directions  
- [ ] Layered training to avoid overfitting in LSTM layers.  
- [ ] Integrate pre-trained language models to enhance semantic understanding.  
- [ ] Implement hierarchical memory structures.  
- [ ] Add attention gating mechanisms.  
- [ ] Optimize memory persistence strategies.  


## Acknowledgments  
This project is inspired by the following works:  
- [Transformer](https://arxiv.org/abs/1706.03762)  
- [DialoGPT](https://github.com/microsoft/DialoGPT)  
- [PLATO](https://arxiv.org/abs/1910.07931)  

Welcome to submit Issues and PRs! 🤝