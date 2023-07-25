#include "qsim/quasistatic_parser.h"

#include <yaml-cpp/yaml.h>

#include <vector>

#include "qsim/get_model_paths.h"

using std::cout;
using std::endl;

std::string ParsePath(std::string file_name) {
  // file name has the format package://package_name/sdf_file_name
  DRAKE_THROW_UNLESS(file_name.substr(0, 10) == "package://");
  file_name = file_name.substr(10);
  auto pos = file_name.find('/');
  auto package_name = file_name.substr(0, pos);
  auto sdf_file_name = file_name.substr(pos + 1);

  return GetPackageMap()[package_name] / sdf_file_name;
}

template <typename T>
void SetValue(const YAML::Node& value, T* value_to_be_set) {
  *value_to_be_set = value.template as<T>();
}

QuasistaticParser::QuasistaticParser(const std::string& q_model_path) {
  auto config = YAML::LoadFile(q_model_path);
  model_directive_path_ =
      ParsePath(config["model_directive"].as<std::string>());

  // Robot stiffness.
  for (const auto& robot : config["robots"]) {
    auto name = robot["name"].as<std::string>();
    Eigen::VectorXd Kp(robot["Kp"].size());
    for (int i = 0; i < robot["Kp"].size(); i++) {
      Kp[i] = robot["Kp"][i].as<double>();
    }
    robot_stiffness_[name] = Kp;
  }

  // Object Sdf paths.
  if (config["objects"]) {
    for (const auto obj : config["objects"]) {
      auto name = obj["name"].as<std::string>();
      object_sdf_paths_[name] = ParsePath(obj["file"].as<std::string>());
    }
  }

  // Simulation Parameters.
  for (const auto& item : config["quasistatic_sim_params"]) {
    auto name = item.first.as<std::string>();
    auto& value = item.second;
    if (name == "h") {
      SetValue(value, &sim_params_.h);
    } else if (name == "gravity") {
      auto& g_vec = value;
      sim_params_.gravity = Eigen::Vector3d(
          g_vec[0].as<double>(), g_vec[1].as<double>(), g_vec[2].as<double>());
    } else if (name == "nd_per_contact") {
      SetValue(value, &sim_params_.nd_per_contact);
    } else if (name == "contact_detection_tolerance") {
      SetValue(value, &sim_params_.contact_detection_tolerance);
    } else if (name == "is_quasi_dynamic") {
      SetValue(value, &sim_params_.is_quasi_dynamic);
    } else if (name == "log_barrier_weight") {
      SetValue(value, &sim_params_.log_barrier_weight);
    } else if (name == "unactuated_mass_scale") {
      SetValue(value, &sim_params_.unactuated_mass_scale);
    } else {
      std::stringstream ss;
      ss << "QuasistaticSimParam " << name << " cannot be set through YAML.";
      throw std::logic_error(ss.str());
    }
  }
}

std::unique_ptr<QuasistaticSimulator> QuasistaticParser::MakeSimulator() const {
  return QuasistaticSimulator::MakeQuasistaticSimulator(
      model_directive_path_, robot_stiffness_, object_sdf_paths_, sim_params_);
}

[[nodiscard]] std::unique_ptr<BatchQuasistaticSimulator>
QuasistaticParser::MakeBatchSimulator() const {
  return std::make_unique<BatchQuasistaticSimulator>(
      model_directive_path_, robot_stiffness_, object_sdf_paths_, sim_params_);
}
