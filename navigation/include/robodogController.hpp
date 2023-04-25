#pragma once

#include <memory>

#include "pathFitter.hpp"
#include "zmq/Subscriber.hpp"
#include "timer.hpp"

// subscriber to HUM_POS
// - when a new pose arrives, push_back, fit and generate new path
// when controller triggered, trim path and data-points behind
// if not initialized (i.e. empty point list), init with current robot pose
// 
class RobodogController {
public:
  RobodogController();
  ~RobodogController();
  void start();
  void stop();
  
  // DEBUGGING METHODS
  std::pair<G2lib::ClothoidList, std::vector<int>> getPath();

private:
  PathFitter pf;
  std::unique_ptr<Subscriber> subHumPos;
  Timer controllerTask;

  void sub_humpos_callback(const char *topic, const char *buf, size_t size, void *data);
  void controller_callback();
  
};