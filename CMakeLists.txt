cmake_minimum_required(VERSION 3.22)
project(embeddings.cpp)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "MSVC Runtime Library" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    add_compile_options("/utf-8")
endif()
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
add_executable(quantize src/quantize.cpp)
add_executable(test_embedding tests/test_embedding.cpp)
add_executable(test_tokenizer tests/test_tokenizer.cpp)

if (EMSCRIPTEN)
    set(EMBEDDINGS_CPP_BUILD_WASM_TOOLS_DEFAULT ON)
else ()
    set(EMBEDDINGS_CPP_BUILD_WASM_TOOLS_DEFAULT OFF)
endif ()
option(EMBEDDINGS_CPP_BUILD_WASM_TOOLS "embeddings.cpp: build wasm browser tools" ${EMBEDDINGS_CPP_BUILD_WASM_TOOLS_DEFAULT})
if (EMBEDDINGS_CPP_BUILD_WASM_TOOLS)
    add_executable(embedding_wasm_bench src/wasm_bench.cpp)
    add_executable(embedding_wasm_model_bench
        src/wasm_model_bench.cpp
        src/base_model.cpp
        src/gte.cpp
        src/utils.cpp
    )
    add_executable(embedding_wasm_model_encode
        src/wasm_model_encode.cpp
        src/base_model.cpp
        src/gte.cpp
        src/utils.cpp
    )
    add_executable(embedding_wasm_model_runner
        src/wasm_model_runner.cpp
        src/base_model.cpp
        src/gte.cpp
        src/utils.cpp
    )

    target_link_libraries(embedding_wasm_bench PRIVATE embeddings)
    target_link_libraries(embedding_wasm_model_bench PRIVATE ggml)
    target_link_libraries(embedding_wasm_model_encode PRIVATE ggml)
    target_link_libraries(embedding_wasm_model_runner PRIVATE ggml)

    if (EMSCRIPTEN)
        set(EMBEDDINGS_WASM_ASSERTIONS "0" CACHE STRING "embeddings.cpp: Emscripten ASSERTIONS value for browser wasm runners")
        set(EMBEDDINGS_WASM_ASYNC_MODE "JSPI" CACHE STRING "embeddings.cpp: Emscripten async mode for browser wasm runners (JSPI or ASYNCIFY)")
        set_property(CACHE EMBEDDINGS_WASM_ASYNC_MODE PROPERTY STRINGS "JSPI" "ASYNCIFY" "NONE")
        target_link_options(embedding_wasm_model_runner PRIVATE
            "SHELL:-sALLOW_MEMORY_GROWTH=1"
            "SHELL:-sASSERTIONS=${EMBEDDINGS_WASM_ASSERTIONS}"
            "SHELL:-sNO_EXIT_RUNTIME=1"
            "SHELL:-sEXPORTED_FUNCTIONS=['_runner_init','_runner_encode','_runner_encode_inline','_runner_last_json','_runner_last_error','_runner_reset']"
            "SHELL:-sEXPORTED_RUNTIME_METHODS=['ccall','UTF8ToString']"
        )
        if (EMBEDDINGS_WASM_ASYNC_MODE STREQUAL "JSPI")
            target_link_options(embedding_wasm_model_runner PRIVATE
                "SHELL:-sJSPI_EXPORTS=['runner_init','runner_encode','runner_encode_inline']"
            )
        elseif (EMBEDDINGS_WASM_ASYNC_MODE STREQUAL "ASYNCIFY")
            target_link_options(embedding_wasm_model_runner PRIVATE
                "SHELL:-sASYNCIFY=1"
            )
        elseif (NOT EMBEDDINGS_WASM_ASYNC_MODE STREQUAL "NONE")
            message(FATAL_ERROR "Unsupported EMBEDDINGS_WASM_ASYNC_MODE: ${EMBEDDINGS_WASM_ASYNC_MODE}")
        endif ()
    endif ()
endif ()

target_link_libraries(embedding_cli PRIVATE embeddings)
target_link_libraries(quantize PRIVATE embeddings)
target_link_libraries(test_embedding PRIVATE embeddings)
target_link_libraries(test_tokenizer PRIVATE embeddings)

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
