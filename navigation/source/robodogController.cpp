#include "robodogController.hpp"
#include "json.hpp"
#include "hardwareglobalinterface.hpp"
#include "control.hpp"

static const std::string HUM_POS_PUBLISHER = "tcp://10.196.80.134:3218";
static const std::string HUM_POS_TOPIC = "HUM_POS";

static const int T_CONTROL = 10;



RobodogController::RobodogController() 
{}

RobodogController::~RobodogController() 
{
  stop();
}

void RobodogController::start() 
{
  HardwareGlobalInterface::getInstance()->robotOn();

  subHumPos = std::make_unique<Subscriber>();
  subHumPos->register_callback(
    [&](const char *topic, const char *buf, size_t size, void *data) {
      sub_humpos_callback(topic, buf, size, data);
    });
  subHumPos->start(HUM_POS_PUBLISHER, HUM_POS_TOPIC);

  controllerTask.registerCallback([&]() {
    controller_callback();
  });
  controllerTask.start(T_CONTROL);
}

void RobodogController::stop()
{
  HardwareGlobalInterface::getInstance()->robotOff();
  
  if (subHumPos)
    subHumPos->stop();
  subHumPos.reset();
  controllerTask.stop();
  //pf.reset(); // TODO
}

void RobodogController::sub_humpos_callback(const char *topic, const char *buf, size_t size, void *data)
{
  nlohmann::json j;

  //std::unique_lock<std::mutex> lock(locDataMtx);

  try{
    j = nlohmann::json::parse(std::string(buf, size));
    // TODO: check if valid??
    bool valid = j.at("valid");
    if (valid) { 
      //double x = j.at("ekf_x");
      //double y = j.at("ekf_y"); 
      //double x = j.at("meas_x_abs");
      //double y = j.at("meas_y_abs");
      double x_rel = j.at("x");
      double y_rel = j.at("y");

      RobotStatus::LocalizationData locData;
      HardwareGlobalInterface::getInstance()->getLocalizationData(locData);
      double x_abs = std::cos(locData.theta)*x_rel - std::sin(locData.theta)*y_rel + locData.x;
      double y_abs = std::sin(locData.theta)*x_rel + std::cos(locData.theta)*y_rel + locData.y;
      pf.push_back(x_abs, y_abs);
    }
  }
  catch(std::exception &e){
    std::cerr << "error parsing humpos data: " << e.what() << std::endl;
  }
}

void RobodogController::controller_callback() {
  double v, omega;

  std::pair<G2lib::ClothoidList, std::vector<int>> path = pf.getPath();
  double s0 = computeControl(T_CONTROL/1000., path.first, v, omega);
  //std::cout << "s0: " << s0 << std::endl;

  if (s0>0) {
    int idx = path.first.findAtS(s0);
    int t0 = path.second[idx];
    pf.trim(t0);
  }

  std::cout << v << " " << omega << std::endl;
  HardwareGlobalInterface::getInstance()->vehicleSafeMove(v, omega);
}

std::pair<G2lib::ClothoidList, std::vector<int>> RobodogController::getPath() {
  return pf.getPath();
}
