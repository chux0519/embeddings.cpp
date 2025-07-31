# 量化格式对比工具使用指南

这个增强版的 `alignment.py` 工具现在支持对同一个模型的多种量化格式进行并行对比。

## 新功能

### 1. 自动量化格式发现
工具会自动扫描 `models/` 目录，发现每个模型的所有可用量化格式：
- `fp16` (半精度浮点)
- `q8_0` (8位量化)
- `q6_k` (6位K量化)
- `q5_k_m` (5位K量化，中等)
- `q5_0` (5位量化)
- `q4_k_m` (4位K量化，中等)
- `q4_k` (4位K量化)
- `q4_0` (4位量化)
- `q3_k_m` (3位K量化，中等)
- `q2_k` (2位K量化)

### 2. 配置文件扩展
在 `MODEL_CONFIGS` 中为每个模型指定支持的量化格式：

```python
MODEL_CONFIGS = {
    'Snowflake/snowflake-arctic-embed-m-v2.0': {
        'pooling': PoolingMethod.CLS, 
        'quantizations': ['fp16', 'q4_k', 'q6_k']
    },
}
```

### 3. 文件命名约定
GGUF 文件应遵循以下命名格式：
```
{model_name}.{quantization}.gguf
```

例如：
- `snowflake-arctic-embed-m-v2.0.fp16.gguf`
- `snowflake-arctic-embed-m-v2.0.q4_k.gguf`
- `snowflake-arctic-embed-m-v2.0.q6_k.gguf`

## 使用方法

### 1. 准备模型文件
将所有量化版本的 GGUF 文件放在 `models/` 目录下：

```
models/
├── snowflake-arctic-embed-m-v2.0.fp16.gguf
├── snowflake-arctic-embed-m-v2.0.q4_k.gguf
└── snowflake-arctic-embed-m-v2.0.q6_k.gguf
```

### 2. 配置模型列表
在 `scripts/models.txt` 中列出要测试的模型：

```
Snowflake/snowflake-arctic-embed-m-v2.0
BAAI/bge-m3
# 其他模型...
```

### 3. 运行对比
```bash
cd scripts
python alignment.py
```

## 输出结果

### CSV 文件
主要结果会保存在带时间戳的 CSV 文件中，包含以下列：
- `repo_name`: 模型仓库名
- `quantization`: 量化格式
- `gguf_file_name`: GGUF 文件名
- `mse`: 均方误差
- `cosine_similarity`: 余弦相似度
- `status`: 测试状态

### Markdown 报告
生成的 Markdown 文件包含：
- **汇总表格**: 所有测试结果的概览
- **分组详情**: 按模型分组显示不同量化格式的对比
- **最佳/最差量化**: 自动标识表现最好和最差的量化格式

### 调试 CSV
详细的调试信息，包含：
- 嵌入向量的前10和后10个值
- 各种归一化信息
- 逐个提示词的相似度

## 示例输出

```
Testing model: Snowflake/snowflake-arctic-embed-m-v2.0
Found quantizations: ['fp16', 'q4_k', 'q6_k']

--- Testing quantization: fp16 ---
Loading C++ model: snowflake-arctic-embed-m-v2.0.fp16.gguf
MSE: 1.23e-05
Cosine Similarity: 0.999876

--- Testing quantization: q4_k ---
Loading C++ model: snowflake-arctic-embed-m-v2.0.q4_k.gguf
MSE: 2.45e-04
Cosine Similarity: 0.998234

--- Testing quantization: q6_k ---
Loading C++ model: snowflake-arctic-embed-m-v2.0.q6_k.gguf
MSE: 8.76e-05
Cosine Similarity: 0.999123

Summary: 3/3 tests passed successfully
Tested 1 unique models with multiple quantizations

Quantization Summary:
  fp16: 1/1 (100.0%)
  q4_k: 1/1 (100.0%)
  q6_k: 1/1 (100.0%)
```

## 分析建议

### 1. 精度 vs 效率权衡
- **fp16**: 最高精度，但文件大小最大
- **q6_k**: 平衡精度和大小，通常是很好的折衷
- **q4_k**: 文件更小，精度适中，适合资源受限环境

### 2. 评估指标
- **余弦相似度 > 0.99**: 表示量化质量很好
- **余弦相似度 > 0.95**: 表示可接受的质量损失
- **余弦相似度 < 0.90**: 可能存在显著的质量损失

### 3. 使用建议
- 对于生产环境，建议使用余弦相似度 > 0.98 的量化格式
- 如果存储空间紧张，可以选择在精度要求范围内文件最小的格式
- 建议测试多个提示词来确保量化的稳定性

## 故障排除

### 常见问题
1. **文件未找到**: 检查 GGUF 文件命名是否符合约定
2. **量化格式不匹配**: 确保配置中的量化格式与实际文件匹配
3. **内存不足**: 大模型的多个量化版本可能消耗大量内存

### 调试技巧
- 查看调试 CSV 文件中的详细数值对比
- 检查嵌入向量的范数是否合理
- 比较不同量化格式的个体相似度分布
