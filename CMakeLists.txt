cmake_minimum_required(VERSION 3.22)
project(embedding)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(ggml)
include_directories(ggml/include)

add_subdirectory(hf_tokenizers)
include_directories(hf_tokenizers/include)


set(EMBEDDING_SOURCES 
    src/bert.cpp
    src/tokenizer.cpp
    src/utils.cpp
)

include_directories(include)
add_executable(embedding main.cpp ${EMBEDDING_SOURCES})
add_executable(test_tokenizer tests/test_tokenizer.cpp ${EMBEDDING_SOURCES})
add_executable(test_embedding tests/test_embedding.cpp ${EMBEDDING_SOURCES})
target_link_libraries(embedding PRIVATE ggml tokenizers_cpp)
target_link_libraries(test_tokenizer PRIVATE ggml tokenizers_cpp)
target_link_libraries(test_embedding PRIVATE ggml tokenizers_cpp)
