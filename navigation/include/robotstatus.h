#ifndef HARDWARESTATUSTYPE_H
#define HARDWARESTATUSTYPE_H

#include <utility>
#include <vector>
#include <mutex>

namespace RobotStatus {

  struct LocalizationData {
    double x = 0;
    double y = 0;
    double theta = 0;
    double timestamp = 0;
  };

  struct OdometryData {
    double x = 0;
    double y = 0;
    double theta = 0;
    double v = 0;
    double omega = 0;
    double timestamp = 0;
  };

  struct LidarDatum {
    double angle = 0;
    double distance = 0;
    double x = 0;
    double y = 0;
    bool isSafe = false;
  };

  struct LidarData{
    std::vector<LidarDatum> datum;
    double lidarTimer = 0;
    double x_offset = 0; // x mounting offset w.r.t the vehicle reference frame
    double y_offset = 0; // y mounting offset w.r.t the vehicle reference frame
    double timestamp = 0;

    void setMountingPosition(double x, double y) {
      x_offset = x;
      y_offset = y;
    }
  };
  
}

#endif // HARDWARESTATUSTYPE_H
