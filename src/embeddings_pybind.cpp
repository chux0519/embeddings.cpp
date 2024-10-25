#include "bert.h"
#include "tokenizer.h"
#include "utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace embeddings {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_C, m) {
  m.doc() = "embeddings.cpp Python bindings";
  py::class_<tokenizers::HFEncoding>(m, "Encoding")
      .def(py::init<>())
      .def_readwrite("ids", &tokenizers::HFEncoding::ids,
                     "Token IDs of the encoding.")
      .def_readwrite("attention_mask", &tokenizers::HFEncoding::attention_mask,
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
}

} // namespace embeddings
