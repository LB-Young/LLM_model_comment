本仓库为经典模型的逐行注释代码库；主要包括Qwen，baichuan，Yi，Llama等

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
    - baichuan2：
        - MLP未作任何处理；将silu传递给transformers.activations的ACT2FN；
    - Yi：
        - MLP未作任何处理；将silu传递给transformers.activations的ACT2FN；
```

2. Attention部分：
```
    - llama：
        - 1、attention部分包含张量并行代码；
        - 2、使用了分组注意力机制，所以在380-381行对k和v进行了复制；
    - Qwen：
        - 1、将QKV的线性变换矩阵合并为self.c_attn，对输入做完线性变换之后，在最后一个维度上split得到Q,K,V;
        - 2、注意力计算部分的点积计算直接调用F.scaled_dot_product_attention；
    - baichuan2：
        - 1、注意力计算部分的点积计算直接调用F.scaled_dot_product_attention；
    - Yi：
        - 1、使用了分组注意力机制，所以在246-253行对k和v进行了复制；
        - 2、代码内部实现了点积计算的流程；
```

3. DecoderLayer部分：
```
    - llama、Qwen、baichuan2、Yi
        - 1、RMS attention resnet RMS MLP resnet
```

4. RMSNorm部分：
```
RMSNorm(a) = a / RMS(a) ⊙ g，其中RMS(a) = √mean(a^2)
```
```
    - llama、Qwen、Yi中self.weight初始化为1矩阵；baichuan2中self.weight初始化为空矩阵；
```