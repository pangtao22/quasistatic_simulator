#include <iostream>
#include <unordered_map>

#include <Eigen/Dense>

#include "batch_quasistatic_simulator.h"
#include "quasistatic_simulator.h"

class QuasistaticParser {
public:
  explicit QuasistaticParser(const std::string &q_model_path);
  void set_sim_params(QuasistaticSimParameters sim_params) {
    sim_params_ = std::move(sim_params);
  };
  const QuasistaticSimParameters &get_sim_params() const {
    return sim_params_;
  };
  [[nodiscard]] std::unique_ptr<QuasistaticSimulator> MakeSimulator() const;
  [[nodiscard]] std::unique_ptr<BatchQuasistaticSimulator>
  MakeBatchSimulator() const;

private:
  std::string model_directive_path_;
  std::unordered_map<std::string, Eigen::VectorXd> robot_stiffness_;
  std::unordered_map<std::string, std::string> object_sdf_paths_;
  QuasistaticSimParameters sim_params_;
};
