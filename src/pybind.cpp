#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "factory.h"

namespace embeddings {

namespace py = pybind11;
using namespace pybind11::literals;

// Trampoline 类，用于让 Python 类可以继承 C++ 的 Embedding 类
class PyEmbedding : public Embedding {
public:
    using Embedding::Embedding;

    TokenizedInput tokenize(const std::string& text, bool add_special_tokens) override {
        PYBIND11_OVERRIDE_PURE(TokenizedInput, Embedding, tokenize, text, add_special_tokens);
    }

    std::vector<TokenizedInput> batch_tokenize(
        const std::vector<std::string>& batch, bool add_special_tokens) override {
        PYBIND11_OVERRIDE_PURE(std::vector<TokenizedInput>, Embedding, batch_tokenize, batch, add_special_tokens);
    }

    std::vector<float> encode(
        const std::string& text, bool normalize, PoolingMethod pooling_method) override {
        PYBIND11_OVERRIDE_PURE(std::vector<float>, Embedding, encode, text, normalize, pooling_method);
    }
    std::vector<std::vector<float>> batch_encode(
        const std::vector<std::string>& batch, bool normalize, PoolingMethod pooling_method) override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::vector<float>>, Embedding, batch_encode, batch, normalize, pooling_method);
    }
};

PYBIND11_MODULE(_C, m) {
    m.doc() = "A unified embedding library for various models.";

    // 绑定 TokenizedInput 结构体
    py::class_<TokenizedInput>(m, "TokenizedInput")
        .def_readwrite("ids", &TokenizedInput::ids, "Token IDs")
        .def_readwrite("attention_mask", &TokenizedInput::attention_mask, "Attention mask")
        .def_readwrite("no_pad_len", &TokenizedInput::no_pad_len, "Length without padding");

    // 绑定 PoolingMethod 枚举
    py::enum_<PoolingMethod>(m, "PoolingMethod")
        .value("MEAN", PoolingMethod::MEAN)
        .value("CLS", PoolingMethod::CLS)
        .export_values();

    // 绑定统一的 Embedding 接口
    py::class_<Embedding, PyEmbedding>(m, "Embedding")
        .def("tokenize", &Embedding::tokenize,
             "Tokenizes a single string into token IDs and attention mask.",
             "text"_a, "add_special_tokens"_a = true)
        .def("batch_tokenize", &Embedding::batch_tokenize,
             "Tokenizes a batch of strings into token IDs and attention masks.",
             "texts"_a, "add_special_tokens"_a = true)
        .def("encode", &Embedding::encode,
             "Encodes a single string into a vector of floats.",
             "text"_a, "normalize"_a = true, "pooling_method"_a = PoolingMethod::MEAN)
        .def("batch_encode", &Embedding::batch_encode,
             "Encodes a batch of strings into a list of float vectors.",
             "texts"_a, "normalize"_a = true, "pooling_method"_a = PoolingMethod::MEAN);

    // 绑定工厂函数，这是用户与之交互的主要方式
    m.def("create_embedding", &create_embedding,
          "Creates a ready-to-use embedding model from a GGUF file.",
          "gguf_path"_a,
          // 告诉 pybind，C++ 返回的 unique_ptr 的所有权将转移给 Python GC
          py::return_value_policy::take_ownership);
}

} // namespace embeddings