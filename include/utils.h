#pragma once

#include <string>
#include <vector>

namespace embeddings {
std::string LoadBytesFromFile(const std::string &path);

void PrintEncodeResult(const std::vector<int> &ids);
} // namespace embeddings