#include <iostream>

#include <drake/common/drake_path.h>
#include <drake/common/find_resource.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/parsing/process_model_directives.h>
#include <drake/multibody/plant/multibody_plant.h>

using std::cout;
using std::endl;

int main() {
  auto builder = drake::systems::DiagramBuilder<double>();
  drake::multibody::MultibodyPlant<double> *plant{nullptr};
  drake::geometry::SceneGraph<double> *scene_graph{nullptr};

  std::tie(plant, scene_graph) =
      drake::multibody::AddMultibodyPlantSceneGraph(&builder, 1e-3);

  auto model_path = drake::FindResourceOrThrow(
      "drake/manipulation/models/ycb/sdf/003_cracker_box.sdf");

  auto parser = drake::multibody::Parser(plant, scene_graph);
  auto model_A = parser.AddModelFromFile(model_path, "A");
  auto model_B = parser.AddModelFromFile(model_path, "B");

  plant->Finalize();
  auto diagram = builder.Build();
  Eigen::VectorXd q0(7);
  q0 << 1, 0, 0, 0, 0, 0, 0;

  Eigen::VectorXd q1(7);
  q1 << 1, 0, 0, 0, -10, -10, -10;

  // Create context and run signed distance queries.
  auto context = diagram->CreateDefaultContext();
  auto& context_plant = plant->GetMyMutableContextFromRoot(context.get());
  auto& context_sg= scene_graph->GetMyMutableContextFromRoot(context.get());
  plant->SetPositions(&context_plant, model_A, q0);
  plant->SetPositions(&context_plant, model_B, q1);
  auto& query_object = scene_graph->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(
      context_sg);
  auto sdps = query_object.ComputeSignedDistancePairwiseClosestPoints(std::numeric_limits<double>::infinity());
  cout << sdps.size() << endl;

  // create new context.
  auto context_new = diagram->CreateDefaultContext();
  auto& context_plant_new = plant->GetMyMutableContextFromRoot(context_new.get());
  auto& context_sg_new= scene_graph->GetMyMutableContextFromRoot(context_new.get());
  plant->SetPositions(&context_plant_new, model_A, q0);
  plant->SetPositions(&context_plant_new, model_B, q1);
  auto& query_object_new = scene_graph->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(
      context_sg_new);
  sdps = query_object_new.ComputeSignedDistancePairwiseClosestPoints(100);
  cout << sdps.size() << endl;



  return 0;
}