#ifndef MAP_MANAGER_HH
#define MAP_MANAGER_HH

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.hpp"


class MapManager
{
public:
  MapManager(int hsize, double resolution, PosedMap const & globalmap);

  void
  setOrigin(double x0, double y0);

  cv::Mat 
  getMapWindow();



  // utility methods
  // get cell (r,c) from coordinates (x,y)
  bool 
  getMapCoordinates(double x, double y, int & r, int & c);

  bool 
  getMapCoordinatesAbs(double x, double y, int & r, int & c);

  // get coordinates (x,y) from cell (r,c)
  bool 
  getWorldCoordinates(int r, int c, double & x, double & y);

  bool 
  getWorldCoordinatesAbs(int r, int c, double & x, double & y);

  void
  getGlobalMapCoordinates(double x, double y, int & r, int & c);

  int
  getSize() { return hsize; }

  double
  getResolution() { return resolution; }

  const PosedMap &
  getGlobalMap() { return global_map; }
  

private:
  PosedMap global_map;

  int hsize;
  double resolution;
  Point2d origin;

  cv::Mat
  getGlobalView(int rmin, int cmin, int rmax, int cmax);

};


#endif
