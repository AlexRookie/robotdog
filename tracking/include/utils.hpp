#pragma once

#include <yaml-cpp/yaml.h>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>

struct Point2d {
  double x, y;
};

struct Pose {
  double x, y, theta;
};

struct Map
{
  double resolution;
  double x0;
  double y0;
  cv::Mat img;

  void map2world(int i, int j, double & x, double & y) const
  {
//    x = (j+0.5)*resolution + x0;
//    y = (i+0.5)*resolution + y0;
    x = (j)*resolution + x0;
    y = (i)*resolution + y0;
  }

  void world2map(double x, double y, int & i, int & j) const
  {
//    i = (y-y0)/resolution+0.5;
//    j = (x-x0)/resolution+0.5;
    i = std::round((y-y0)/resolution);
    j = std::round((x-x0)/resolution);
  }
};


struct PosedMap
{
  cv::Mat map;
  Pose pose;
};


inline
bool loadMapFile(std::string filename, PosedMap & map)
{
  std::string imgUrl;
  double resolution;
  int negate;
  double occThresh;
  double freeThresh;

  try 
  {
    YAML::Node mapConf = YAML::LoadFile(filename);

    imgUrl = mapConf["image"].as<std::string>();
    resolution = mapConf["resolution"].as<double>();
    negate = mapConf["negate"].as<int>();
    occThresh = mapConf["occupied_thresh"].as<double>();
    freeThresh = mapConf["free_thresh"].as<double>();
    std::vector<double> origin = mapConf["origin"].as<std::vector<double>>();

    std::experimental::filesystem::path mapConfPath(filename);
    mapConfPath.remove_filename();
    imgUrl = (mapConfPath/imgUrl).string();

    //map.resolution = resolution;
    map.pose.x = origin[0];
    map.pose.y = origin[1];
    map.pose.theta = origin[2]; 
    map.map = cv::imread(imgUrl.c_str(), cv::IMREAD_GRAYSCALE);

    cv::threshold(map.map, map.map, (1-freeThresh)*255, 255, 
      negate ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);

    //map.map = -map.map + cv::Scalar(255);

    cv::flip(map.map, map.map, 0);
  }
  catch (std::exception & ex)
  {
    std::cerr << ex.what() << std::endl;
    return false;
  }

  //flip(origImage, image, 0); // flip image around x axis: the origin of the map is at the bottom left corner
  return true;
}
