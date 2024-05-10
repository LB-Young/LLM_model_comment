本仓库为经典模型的逐行注释代码库；主要包括Qwen，Baichuan2，Yi，Llama等
### 代码来源
    - llama：transformers仓库
    - Qwen：huggingface
    - Baichuan2：huggingface
    - Yi：huggingface

### 进度：
    - llama源码主体部分已经注释完成；

### 代码对比：

1. MLP代码部分：
```
self.down(self.up(x) * self.silu(self.gate(x)))        
Swish(x) = x*sigmoid(ßx)   GLU(x) = sigmoid(W1x+b)⊗(Vx+c)   SwiGLU(x) = Swish(W1x+b)⊗(Vx+c)
```
```
    - llama：
        - MLP部分增加了张量并行的策略；将silu传递给transformers.activations的ACT2FN；
    - Qwen：
        - MLP部分第一层的输出不是intermediate_size，而是intermediate_size//2；直接调用F.silu;
    - Baichuan2：
        - MLP未作任何处理；将silu传递给transformers.activations的ACT2FN；
    - Yi：
        - MLP未作任何处理；将silu传递给transformers.activations的ACT2FN；
```

2. Attention部分：
```
    - llama：
        - 1、init主要定义了四个线性层，并且初始化了rope；
        - 2、attention部分包含张量并行代码；
        - 3、使用了分组注意力机制，所以在380-381行对k和v进行了复制；
    - Qwen：
        - 1、将QKV的线性变换矩阵合并为self.c_attn，对输入做完线性变换之后，在最后一个维度上split得到Q,K,V;
        - 2、注意力计算部分的点积计算直接调用F.scaled_dot_product_attention；
    - Baichuan2：
        - 1、注意力计算部分的点积计算直接调用F.scaled_dot_product_attention；
    - Yi：
        - 1、使用了分组注意力机制，所以在246-253行对k和v进行了复制；
        - 2、代码内部实现了点积计算的流程；
```

3. DecoderLayer部分：
```
    - llama、Qwen、Baichuan2、Yi
        - 1、RMS attention resnet RMS MLP resnet
```

4. RMSNorm部分：
```
RMSNorm(a) = a / RMS(a) ⊙ g，其中RMS(a) = √mean(a^2)
```
```
    - llama、Qwen、Yi中self.weight初始化为1矩阵；Baichuan2中self.weight初始化为空矩阵；
```

5. FlashAttention部分：
```
    - llama：
        - 1、没有attention中的张量并行；没有分组注意力；
```

6. ROPE部分：
```
    - llama:
        - 1、非常优雅，没有乱七八糟的操作，定义了inv_freq和t之后einsum得到emb，接着做了cos和sin处理；
        - 2、rotate_half中交换元素顺序，没有按照图片中的位置交换，而是使用切片，前后半区的交换；
```

7. PreTrainedModel部分：
```
    - llama:
        - 1、初始化模型的线性层权重(默认0均值，0.02方差)、偏差（默认0）
        - 2、embedding层权重（默认0均值，0.02方差）、pad向量全0；
```