#include "hardwareglobalinterface.hpp"

#include <json.hpp>
#include <chrono>
//#include "utils.hpp"

struct Point2d {
  Point2d() {}
  Point2d(double x, double y): x(x), y(y) {}

  // friend bool operator==(Point2d const & p1, Point2d const & p2) { return p1.x==p2.x && p1.y==p2.y; }
  // friend bool operator!=(Point2d const & p1, Point2d const & p2) { return !(p1==p2); }

  double x, y;
};

struct EulerAngles {
  double roll, pitch, yaw;
};

struct Quaternion {
  double x, y, z, w;
};




static inline double isLeft(Point2d const & P0, Point2d const & P1, Point2d const & P2)
{
  return ((P1.x - P0.x) * (P2.y - P0.y) -(P2.x - P0.x) * (P1.y - P0.y));
}

static bool contain(Point2d const & point, std::vector<Point2d> const & vertices)
{
  int wn = 0; // the winding number counter

  // loop through all edges of the polygon
  for (int i=1; i<vertices.size(); ++i)
  {
    // edge from V[i] to V[i+1]
    if (vertices.at(i-1).y <= point.y)
    {
      // start y <= P.y
      if (vertices.at(i).y > point.y) // an upward crossing
        if (isLeft(vertices.at(i-1), vertices.at(i), point) > 0) // P left of edge
          ++wn; // have a valid up intersect
    }
    else
    {
      // start y > P.y (no test needed)
      if (vertices.at(i).y <= point.y) // a downward crossing
        if (isLeft(vertices.at(i-1), vertices.at(i), point) < 0) // P right of edge
          --wn; // have a valid down intersect
    }
  }
  return wn != 0;
}

static inline EulerAngles ToEulerAngles(Quaternion const & q) {
  EulerAngles angles;

  double test = q.x*q.y + q.z*q.w;
  if (test > 0.499) { // singularity at north pole
    angles.roll = 2 * std::atan2(q.x,q.w);
    angles.pitch = M_PI/2;
    angles.yaw = 0;
    return angles;
  }
  if (test < -0.499) { // singularity at south pole
    angles.roll = -2 * std::atan2(q.x,q.w);
    angles.pitch = - M_PI/2;
    angles.yaw = 0;
    return angles;
  }
  double sqx = q.x*q.x;
  double sqy = q.y*q.y;
  double sqz = q.z*q.z;
  angles.roll = std::atan2(2*q.y*q.w-2*q.x*q.z , 1 - 2*sqy - 2*sqz);
  angles.pitch = std::asin(2*test);
  angles.yaw = std::atan2(2*q.x*q.w-2*q.y*q.z , 1 - 2*sqx - 2*sqz);

  return angles;
}



std::unique_ptr<HardwareGlobalInterface> HardwareGlobalInterface::instance;


HardwareGlobalInterface * HardwareGlobalInterface::getInstance() {
  if (!instance) throw std::runtime_error("Robot not initialized");
  return instance.get();
}

void HardwareGlobalInterface::initialize(HardwareParameters const & hp) {
  instance = std::unique_ptr<HardwareGlobalInterface>(new HardwareGlobalInterface(hp));
    
  instance->lidarData.setMountingPosition(0, 0);
  
  instance->subLoc.reset(new Subscriber());
  instance->subLoc->register_callback(
    [&](const char *topic, const char *buf, size_t size, void *data) {
      instance->sub_loc_callback(topic, buf, size, data);
    });
  instance->subLoc->start(hp.localizationPublisher, hp.localizationTopic);

  instance->subOdom.reset(new Subscriber());
  instance->subOdom->register_callback(
    [&](const char *topic, const char *buf, size_t size, void *data) {
      instance->sub_odom_callback(topic, buf, size, data);
    });
  instance->subOdom->start(hp.odomPublisher, hp.odomTopic);

  instance->subLidar.reset(new Subscriber());
  instance->subLidar->register_callback(
    [&](const char *topic, const char *buf, size_t size, void *data) {
      instance->sub_lidar_callback(topic, buf, size, data);
    });
  instance->subLidar->start(hp.lidarPublisher, hp.lidarTopic);

  instance->reqHW.reset(new RequesterSimple(hp.hardwareServer));
}



HardwareParameters HardwareGlobalInterface::getParams() {
  return params;
}

bool HardwareGlobalInterface::getLocalizationData(RobotStatus::LocalizationData &locData){
  std::unique_lock<std::mutex> lock(locDataMtx);
  double currTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  locData = this->locData;
  if (currTime - locData.timestamp < LOC_TIMEOUT) {
    return true;
  } else {
    return false;
  }
}

bool HardwareGlobalInterface::getOdomData(RobotStatus::OdometryData &odomData) {
  std::unique_lock<std::mutex> lock(odomDataMTX);
  double currTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  odomData = this->odomData;
  if (currTime - odomData.timestamp < ODOM_TIMEOUT) {
    return true;
  } else {
    return false;
  }
}

bool HardwareGlobalInterface::getLidarData(RobotStatus::LidarData &lidarData){
  std::unique_lock<std::mutex> lock(lidarDataMtx);
  double currTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  lidarData = this->lidarData;
  if (currTime - lidarData.timestamp < LIDAR_TIMEOUT) {
    return true;
  } else {
    return false;
  }
}




HardwareGlobalInterface::~HardwareGlobalInterface()
{
  subLoc->stop();
  subOdom->stop();
  subLidar->stop();
}


HardwareGlobalInterface::HardwareGlobalInterface()
{}

HardwareGlobalInterface::HardwareGlobalInterface(HardwareParameters const & hp)
{
  params = hp;
}

void HardwareGlobalInterface::sub_loc_callback(const char *topic, const char *buf, size_t size, void *data)
{
  nlohmann::json j;

  std::unique_lock<std::mutex> lock(locDataMtx);

  try{
    j = nlohmann::json::parse(std::string(buf, size));
    locData.x = j.at("loc_data").at("x");
    locData.y = j.at("loc_data").at("y");
    locData.theta = j.at("loc_data").at("theta");
    locData.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }
  catch(std::exception &e){
    std::cerr << "error parsing loc data: " << e.what() << std::endl;
  }
}

void HardwareGlobalInterface::sub_odom_callback(const char *topic, const char *buf, size_t size, void *data)
{
  std::unique_lock<std::mutex> lock(odomDataMTX);

  nlohmann::json j;
  try {
    j = nlohmann::json::parse(std::string(buf, size));
    double x, y, theta;
    double v, omega;
    long long int ts;

    odomData.x = j["pose"]["pose"]["position"]["x"]; 
    odomData.y = j["pose"]["pose"]["position"]["y"]; 

    Quaternion q;
    q.x = j["pose"]["pose"]["orientation"]["x"];
    q.y = j["pose"]["pose"]["orientation"]["y"];
    q.z = j["pose"]["pose"]["orientation"]["z"];
    q.w = j["pose"]["pose"]["orientation"]["w"];
    EulerAngles angles = ToEulerAngles(q);
    odomData.theta = angles.yaw;

    odomData.v = j["twist"]["twist"]["linear"]["x"];
    odomData.omega = j["twist"]["twist"]["angular"]["z"];

    odomData.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  } catch(std::exception &e) {
    std::cerr << "error parsing odom data: " << e.what() << std::endl;
  }
}

void HardwareGlobalInterface::sub_lidar_callback(const char *topic, const char *buf, size_t size, void *data)
{
  std::unique_lock<std::mutex> lock(lidarDataMtx);

  nlohmann::json j;
  int count = 0;

  try {
    j = nlohmann::json::parse(std::string(buf, size));

    //std::cout << j.dump() << std::endl;
    int size = j.at("size");
    double val;
    double angle = 0;
    double inc = 360.0/((double)(size));
    double angle_offset = 0;

    lidarData.datum.clear();

    //std::cout << j.dump() << std::endl;

    for(int i=0; i<size; i++) {
      val = j.at("data")[i];

      angle = count*inc*M_PI/180.0 + angle_offset;
      count++;

      if(val < 0.05){
        continue;
      }

      //Normalize the angle within 0/2Pi
      angle = atan2(sin(angle), cos(angle));

      RobotStatus::LidarDatum datumLidar;
      datumLidar.angle = angle;
      datumLidar.distance = val;
      datumLidar.x = val*cos(angle) + lidarData.x_offset;
      datumLidar.y = val*sin(angle) + lidarData.y_offset;
      datumLidar.isSafe = true;
      lidarData.datum.push_back(datumLidar);
    }

    lidarData.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  }
  catch(std::exception &e){
    std::cerr << "error parsing lidar data: " << e.what() << std::endl;
  }
}


void HardwareGlobalInterface::powerEnable(bool val){
  RequesterSimple::status_t req_status;
  nlohmann::json j_req;
  if(val){
    j_req["cmd"] = std::string("set_power_enable");
    j_req["enable"] = true;
    std::string response;
    reqHW->request(j_req.dump(),response,req_status);
    std::cout << j_req.dump()<<std::endl;
    if(req_status==RequesterSimple::status_t::STATUS_OK){
      try{nlohmann::json j_resp = nlohmann::json::parse(response);
        std::cout << j_resp.dump()<<std::endl;
        if(j_resp.at("ack") == "true"){
        }else{

        }
      }catch(std::exception &e){
        std::cerr << "POWER_ENABLE TRUE" << std::endl;
        std::cerr << "\"" << response << "\"" << std::endl;
        std::cerr<<e.what()<<std::endl;
      }
    }else{

    }
  }else{
    j_req["cmd"] = std::string("set_power_enable");
    j_req["enable"] = false;
    std::string response;
    reqHW->request(j_req.dump(),response,req_status);
    if(req_status==RequesterSimple::status_t::STATUS_OK){
      try{nlohmann::json j_resp = nlohmann::json::parse(response);
        if(j_resp.at("ack") == "true"){

        }else{

        }
      }catch(std::exception &e){
        std::cerr << "POWER_ENABLE FALSE" << std::endl;
        std::cerr << "\"" << response << "\"" << std::endl;
        std::cerr<<e.what()<<std::endl;
      }
    }else{

    }
  }
}



void HardwareGlobalInterface::robotOn()
{
  powerEnable(true);
  setDeviceMode(5);
}

void HardwareGlobalInterface::robotOff()
{
  powerEnable(false);
}

void HardwareGlobalInterface::setDeviceMode(int deviceMode){
  RequesterSimple::status_t req_status;
  nlohmann::json j_req;
  j_req["cmd"] = std::string("set_device_mode");
  j_req["device_mode"] = deviceMode;
  std::string response;
  reqHW->request(j_req.dump(),response,req_status);
  std::cout << j_req.dump()<<std::endl;
  if(req_status==RequesterSimple::status_t::STATUS_OK){
    try{nlohmann::json j_resp = nlohmann::json::parse(response);
      std::cout << j_resp.dump()<<std::endl;
      if(j_resp.at("ack") == "true"){

      }else{

      }
    }catch(std::exception &e){
      std::cerr << "SET_DEVICE_MODE" << std::endl;
      std::cerr << "\"" << response << "\"" << std::endl;
      std::cerr<<e.what()<<std::endl;
    }
  }else{

  }
}


void HardwareGlobalInterface::vehicleMove(float vel, float omega){

  RequesterSimple::status_t req_status;
  nlohmann::json j_req;
  j_req["cmd"] = std::string("move");
  j_req["speed"] = vel;
  j_req["omega"] = omega;



  std::string response;
  reqHW->request(j_req.dump(),response,req_status);
  if(req_status==RequesterSimple::status_t::STATUS_OK){
    try{nlohmann::json j_resp = nlohmann::json::parse(response);

      if(j_resp.at("ack") == "true"){

      }else{

      }
    }catch(std::exception &e){
      std::cerr << "VEHICLE_MOVE" << std::endl;
      std::cerr << "\"" << response << "\"" << std::endl;
      std::cerr<<e.what()<<std::endl;
    }
  }else{

  }
}

void HardwareGlobalInterface::vehicleSafeMove(float vel, float omega)
{
  double ha = params.ha;
  double hb = params.hb;
  double safetyWidth = params.safetyWidth;
  double criticalDistance = params.criticalDistance;
  double width = params.width;

  //First of all check possible collisions
  std::vector<RobotStatus::LocalizationData> futurePoses;
  futurePoses.clear();
  RobotStatus::LocalizationData actPose;
  actPose.x = 0;
  actPose.y = 0;
  actPose.theta = 0;
  actPose.timestamp = 0;

  futurePoses.push_back(actPose);
  double dt = 0.5;
  double distanceStar = 0;

  for(double t=0;t<=2.5;t=t+dt){
    double theta_end = futurePoses.back().theta+omega*dt;
    double dx = vel*dt*cos(futurePoses.back().theta+omega*dt/2.0);
    double dy = vel*dt*sin(futurePoses.back().theta+omega*dt/2.0);

    double xk = futurePoses.back().x + dx;
    double yk = futurePoses.back().y  + dy;

    if(t==1.5){
      if(futurePoses.back().x<0){
        distanceStar = sqrt(pow(xk-hb*cos(theta_end),2)+pow(yk-hb*sin(theta_end),2))-hb;
      }else{
        distanceStar = sqrt(pow(xk,2)+pow(yk,2))+ha;
      }
    }

    RobotStatus::LocalizationData nextPose;
    nextPose.x = xk;
    nextPose.y = yk;
    nextPose.theta = theta_end;
    nextPose.timestamp = futurePoses.back().timestamp + dt;

    futurePoses.push_back(nextPose);
  }

  //Now check possible collisions
  double numberZoneGreen = 0;
  double numberZoneYellow = 0;
  double numberZoneRed = 0;
  double numberZoneCritical = 0;
  double closestDistance = 99999;

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////CHECK WITH THE LIDAR//////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  RobotStatus::LidarData lidarScans;
  HardwareGlobalInterface::getInstance()->getLidarData(lidarScans);
  for (std::vector<RobotStatus::LidarDatum>::iterator itLidar = lidarScans.datum.begin() ; itLidar != lidarScans.datum.end(); ++itLidar){
    //Check  if it is inside the critical zone
    Point2d p1,p2,p3,p4;
    if(futurePoses.at(1).x>0){
      p1 = Point2d(0, -safetyWidth);
      p2 = Point2d(criticalDistance+ha, -safetyWidth);
      p3 = Point2d(criticalDistance+ha, safetyWidth);
      p4 = Point2d(0, safetyWidth);

      std::vector<Point2d> polygonVertex;
      polygonVertex.push_back(p1);
      polygonVertex.push_back(p2);
      polygonVertex.push_back(p3);
      polygonVertex.push_back(p4);
      polygonVertex.push_back(p1);

      Point2d lidarPoint;
      lidarPoint.x = ((*itLidar).x);
      lidarPoint.y = ((*itLidar).y);
      if(contain(lidarPoint,polygonVertex)){
        (*itLidar).isSafe = false;
        numberZoneCritical++;
        continue;
      }

    }else if(futurePoses.at(1).x<0){
      p1 = Point2d(-hb, -safetyWidth);
      p2 = Point2d(-criticalDistance-hb, -safetyWidth);
      p3 = Point2d(-criticalDistance-hb, safetyWidth);
      p4 = Point2d(-hb, safetyWidth);
      std::vector<Point2d> polygonVertex;
      polygonVertex.push_back(p1);
      polygonVertex.push_back(p2);
      polygonVertex.push_back(p3);
      polygonVertex.push_back(p4);
      polygonVertex.push_back(p1);

      Point2d lidarPoint;
      lidarPoint.x = ((*itLidar).x);
      lidarPoint.y = ((*itLidar).y);
      if(contain(lidarPoint,polygonVertex)){
        (*itLidar).isSafe = false;
        numberZoneCritical++;
        //I associated the lidar Point to the critical zone, go to the next one
        continue;
      }

    }

    if(futurePoses.at(1).y<-0.05){
      p1 = Point2d(ha, -width/2);
      p2 = Point2d(ha,-criticalDistance- width/2);
      p3 = Point2d(-hb, -criticalDistance- width/2);
      p4 = Point2d(-hb, -width/2);
      std::vector<Point2d> polygonVertex;
      polygonVertex.push_back(p1);
      polygonVertex.push_back(p2);
      polygonVertex.push_back(p3);
      polygonVertex.push_back(p4);
      polygonVertex.push_back(p1);

      Point2d lidarPoint;
      lidarPoint.x = ((*itLidar).x);
      lidarPoint.y = ((*itLidar).y);
      if(contain(lidarPoint,polygonVertex)){
        (*itLidar).isSafe = false;
        numberZoneCritical++;
        //I associated the lidar Point to the critical zone, go to the next one
        continue;
      }
    }else if(futurePoses.at(1).y>0.05){
      p1 = Point2d(ha, width/2);
      p2 = Point2d(ha, criticalDistance + width/2);
      p3 = Point2d(-hb, criticalDistance + width/2);
      p4 = Point2d(-hb, -width/2);
      std::vector<Point2d> polygonVertex;
      polygonVertex.push_back(p1);
      polygonVertex.push_back(p2);
      polygonVertex.push_back(p3);
      polygonVertex.push_back(p4);
      polygonVertex.push_back(p1);

      Point2d lidarPoint;
      lidarPoint.x = ((*itLidar).x);
      lidarPoint.y = ((*itLidar).y);
      if(contain(lidarPoint,polygonVertex)){
        (*itLidar).isSafe = false;
        numberZoneCritical++;
        //I associated the lidar Point to the critical zone, go to the next one
        continue;
      }

    }

    int zoneCount = 1;
    for (std::vector<RobotStatus::LocalizationData>::iterator itPose = futurePoses.begin() ; itPose != futurePoses.end(); ++itPose){

      RobotStatus::LocalizationData p = (*itPose);

      Point2d p1 = Point2d(p.x + ha*cos(p.theta) + safetyWidth*cos(p.theta+M_PI_2), p.y + ha*sin(p.theta) + safetyWidth*sin(p.theta+M_PI_2));
      Point2d p2 = Point2d(p.x + ha*cos(p.theta) - safetyWidth*cos(p.theta+M_PI_2), p.y + ha*sin(p.theta) - safetyWidth*sin(p.theta+M_PI_2));
      Point2d p3 = Point2d(p.x - hb*cos(p.theta) - safetyWidth*cos(p.theta+M_PI_2), p.y - hb*sin(p.theta) - safetyWidth*sin(p.theta+M_PI_2));
      Point2d p4 = Point2d(p.x - hb*cos(p.theta) + safetyWidth*cos(p.theta+M_PI_2), p.y - hb*sin(p.theta) + safetyWidth*sin(p.theta+M_PI_2));

      std::vector<Point2d> polygonVertex;
      polygonVertex.push_back(p1);
      polygonVertex.push_back(p2);
      polygonVertex.push_back(p3);
      polygonVertex.push_back(p4);
      polygonVertex.push_back(p1);

      Point2d lidarPoint;
      lidarPoint.x = ((*itLidar).x);
      lidarPoint.y = ((*itLidar).y);
      if(contain(lidarPoint,polygonVertex)){

        double tmp_distance = sqrt(pow((*itLidar).x,2)+pow((*itLidar).y,2));
        if(tmp_distance<closestDistance){
          closestDistance = tmp_distance;
        }

        (*itLidar).isSafe = false;
        if(zoneCount==1 || zoneCount==2){
          numberZoneRed++;
        }else if (zoneCount==3||zoneCount==4){
          numberZoneYellow++;
        }else if(zoneCount==5||zoneCount==6){
          numberZoneGreen++;
        }

        //As soon as the lidar point is associated to a region -> I continue the loop
        break;
      }
      zoneCount++;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////TAKE A DECISION////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  double brakingCoefficient = 1.0;
  if(numberZoneCritical!=0){
    brakingCoefficient = 0.0;
  }else if(numberZoneRed!=0){
    brakingCoefficient = fabs(closestDistance/distanceStar);
  }else if (numberZoneYellow!=0){
    brakingCoefficient = fabs(closestDistance/distanceStar);
  }else if (numberZoneGreen!=0){
    brakingCoefficient = 1.0;
  }else{
    brakingCoefficient = 1.0;
  }

  lastInputVel = vel;
  lastInputOmega = omega;

  if (brakingCoefficient>this->brakingCoefficient) {
    //Ho strada libera per cui voglio accelerare
    brakingCoefficient = this->brakingCoefficient+0.01;
  }
  this->brakingCoefficient = brakingCoefficient;

  //Move the vehicle
  vehicleMove(vel*brakingCoefficient,omega*brakingCoefficient);
}

