#ifndef HARDWAREGLOBALINTERFACE_H
#define HARDWAREGLOBALINTERFACE_H

#define LOC_REALSENSE

#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "single_thread.hpp"
#include "hardwareparameters.hpp"

#include "zmq/Subscriber.hpp"
#include "zmq/RequesterSimple.hpp"
#include "robotstatus.h"

#define LOC_TIMEOUT   50
#define ODOM_TIMEOUT  50
#define LIDAR_TIMEOUT 200

class HardwareGlobalInterface
{
public:

  static HardwareGlobalInterface * getInstance();

  static void initialize(HardwareParameters const & hp);
  
  ~HardwareGlobalInterface();

  HardwareParameters getParams();

  bool getLocalizationData(RobotStatus::LocalizationData &locData);
  bool getOdomData(RobotStatus::OdometryData &odomData);
  bool getLidarData(RobotStatus::LidarData &lidarData);

  void robotOn();
  void robotOff();
  void vehicleMove(float vel, float omega);
  //Safely move the vehicle since it ckecks for possible collisions
  void vehicleSafeMove(float vel, float omega);
  
private:

  HardwareGlobalInterface();
  HardwareGlobalInterface(HardwareParameters const & hp);


  void sub_loc_callback(const char *topic, const char *buf, size_t size, void *data);
  void sub_odom_callback(const char *topic, const char *buf, size_t size, void *data);
  void sub_lidar_callback(const char *topic, const char *buf, size_t size, void *data);
  
  void setDeviceMode(int deviceMode);
  void powerEnable(bool val);

  static std::unique_ptr<HardwareGlobalInterface> instance;

  std::mutex locDataMtx;
  RobotStatus::LocalizationData locData;
  std::unique_ptr<Subscriber> subLoc;

  std::mutex odomDataMTX;
  RobotStatus::OdometryData odomData;
  std::unique_ptr<Subscriber> subOdom;

  std::mutex lidarDataMtx;
  RobotStatus::LidarData lidarData;
  std::unique_ptr<Subscriber> subLidar;

  std::unique_ptr<RequesterSimple> reqHW;

  HardwareParameters params;
  
  double lastInputVel = 0;
  double lastInputOmega = 0;
  double brakingCoefficient = 1.0;
};

#endif // HARDWAREGLOBALINTERFACE_H
