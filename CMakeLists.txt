cmake_minimum_required(VERSION 3.22)
project(embeddings.cpp)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "MSVC Runtime Library" FORCE)

add_subdirectory(ggml)
include_directories(ggml/include)

add_subdirectory(hf_tokenizers)
include_directories(hf_tokenizers/include)

include_directories(src)

set(EMBEDDING_SOURCES
    src/base_model.cpp
    src/gte.cpp
    src/jina_bert.cpp
    src/bert.cpp
    src/tokenizer.cpp
    src/utils.cpp
    src/factory.cpp
)
add_library(embeddings STATIC ${EMBEDDING_SOURCES})
target_link_libraries(embeddings PUBLIC ggml tokenizers_cpp)

add_executable(embedding_cli src/cli.cpp)
add_executable(test_tokenizer tests/test_tokenizer.cpp)
add_executable(test_embedding tests/test_embedding.cpp)
add_executable(quantize src/quantize.cpp)

target_link_libraries(embedding_cli PRIVATE embeddings)
target_link_libraries(test_tokenizer PRIVATE embeddings)
target_link_libraries(test_embedding PRIVATE embeddings)
target_link_libraries(quantize PRIVATE embeddings)

option(EMBEDDINGS_CPP_ENABLE_PYBIND "embeddings.cpp: enable python binding" OFF)
if (EMBEDDINGS_CPP_ENABLE_PYBIND)
    set_target_properties(embeddings ggml tokenizers_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    add_subdirectory(pybind11)
    pybind11_add_module(_C src/pybind.cpp)
    target_link_libraries(_C PRIVATE embeddings)
endif ()

# stub
add_custom_target(stub
    pybind11-stubgen embeddings_cpp -o .
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)