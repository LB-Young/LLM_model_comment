"""
矩阵乘法
    - a * b : 对位相乘(Hadamard product)
    - torch.matmul(a, b) == a @ b : 矩阵乘和内积 支持 broadcast
        - torch.dot(): 1d 不支持 broadcast  内积（元素对位相乘求和）
        - torch.mm(): 2d 不支持 broadcast  矩阵乘   第一个矩阵的列一定等于第二个矩阵的行
        - torch.bmm(): 3d 不支持 broadcast  矩阵乘  在torch.mm()的基础上加上一个batch维度，并且两个矩阵的batch
        维度相同
"""