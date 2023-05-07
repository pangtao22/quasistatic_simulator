#include <filesystem>
#include <unordered_map>


std::filesystem::path GetQsimModelsPath();
std::filesystem::path GetRoboticsUtilitiesModelsPath();
std::unordered_map<std::string, std::filesystem::path> GetPackageMap();
