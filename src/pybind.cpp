#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "bert.h"
#include "jina_bert.h"
#include "tokenizer.h"

namespace embeddings {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_C, m) {
  m.doc() = "embeddings.cpp Python bindings";
  py::class_<Encoding>(m, "Encoding")
      .def(py::init<>())
      .def_readwrite("ids", &Encoding::ids, "Token IDs of the encoding.")
      .def_readwrite("attention_mask", &Encoding::attention_mask,
                     "Attention mask for the encoding.");

  py::bind_vector<tokens>(m, "Tokens");
  py::bind_vector<tokens_batch>(m, "TokensBatch");

  py::class_<Tokenizer>(m, "Tokenizer")
      .def(py::init<const std::string &>(), "path"_a)

      .def("encode", &Tokenizer::Encode, "Encodes a single string into tokens.",
           "text"_a, "add_special_tokens"_a = true)

      .def("encode_batch", &Tokenizer::EncodeBatch,
           "Encodes a batch of strings into tokens.", "texts"_a,
           "add_special_tokens"_a = true)

      .def("decode", &Tokenizer::Decode, "Decodes tokens into a string.",
           "tokens"_a, "skip_special_tokens"_a = true);

  py::class_<Embedding>(m, "Embedding")
      .def(py::init<const std::string &, const std::string &>(),
           "hf_token_json"_a, "gguf_model"_a)
      .def("encode", &Embedding::Encode,
           "Encodes a single string into a vector of floats.", "text"_a,
           "normalize"_a = true, "pooling_method"_a = 0)
      .def("batch_encode", &Embedding::BatchEncode,
           "Encodes a batch of strings into a list of float vectors.",
           "texts"_a, "normalize"_a = true, "pooling_method"_a = 0);
}

}  // namespace embeddings
