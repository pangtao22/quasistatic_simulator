#include "get_model_paths.h"

#include "drake/common/drake_path.h"

using std::filesystem::current_path;
using std::filesystem::path;

std::filesystem::path GetQsimModelsPath() {
  static auto file_path = path(__FILE__);
  static auto q_sim_models_path =
      file_path.parent_path() / path("../models");
  return q_sim_models_path;
}

std::filesystem::path GetRoboticsUtilitiesModelsPath() {
  static auto file_path = path(__FILE__);
  static auto py_package_path =
      file_path.parent_path() / path("../..");
  static auto robo_util_models_path = py_package_path /
                                      path("robotics_utilities_pang") /
                                      path("robotics_utilities/models");
  return robo_util_models_path;
}

std::unordered_map<std::string, std::filesystem::path> GetPackageMap() {
  auto drake_path = path(drake::MaybeGetDrakePath().value());

  static std::unordered_map<std::string, std::filesystem::path> package_map = {
      {"quasistatic_simulator", GetQsimModelsPath()},
      {"drake_manipulation_models", drake_path / "manipulation" / "models"}};

  return package_map;
}
