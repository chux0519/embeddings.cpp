cmake_minimum_required(VERSION 3.22)
project(potasms_cpp)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(ggml)
include_directories(ggml/include)

add_subdirectory(tokenizers-cpp)
include_directories(tokenizers-cpp/include)


include_directories(include)
add_executable(potasms_cpp 
    main.cpp src/bert.cpp src/tokenizer.cpp src/utils.cpp
)
target_link_libraries(potasms_cpp PRIVATE ggml tokenizers_cpp)
