## 模型准备

> pip install -r scripts/requirements.txt
>
> python scripts/convert.py BAAI/bge-m3 ./models/bge-m3.fp16.gguf f16
>
> python scripts/convert.py BAAI/bge-base-zh-v1.5 ./models/bge-base-zh-v1.5.fp16.gguf f16
>
> python scripts/convert.py shibing624/text2vec-base-multilingual ./models/text2vec-base-multilingual.fp16.gguf f16

## 运行测试

> ./build/test_embedding

## 运行笔记本

运行之前，安装 embeddings.cpp

> pip install .
> cd build && make stub


## 编译

### mac(arm)

> cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DEMBEDDINGS_CPP_ENABLE_PYBIND=ON ..
