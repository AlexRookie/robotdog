#include "MapManager.hpp"

#include <limits>

static const uint8_t EMPTY = 0;
static const uint8_t FULL  = 255;


template <typename T>
static inline int sgn(T const & v)
{
  return (v-T(0))>0 - (v-T(0))<0;
}

MapManager::MapManager(int hsize, double resolution, PosedMap const & globalmap):
  hsize(hsize), resolution(resolution), origin({0.,0}), global_map(globalmap)
{}

void
MapManager::setOrigin(double x0, double y0) {
  origin = {x0, y0};
}


cv::Mat
MapManager::getGlobalView(int rmin, int cmin, int rmax, int cmax)
{
  //dbg.width = dbg.height = 0;

  assert(rmax-rmin==2*hsize && cmax-cmin==2*hsize);
  int size = 2*hsize+1;
  cv::Mat res(size, size, CV_8U, EMPTY);

  int grows = global_map.map.rows;
  int gcols = global_map.map.cols;

  if (rmin>=grows || rmax<0)
    return res;
  if (cmin>=gcols || cmax<0)
    return res;

  cv::Rect window(0, 0, size, size);
  if (rmax>=grows)
  {
    window.height -= rmax-grows+1;
  }
  if (cmax>=gcols)
  {
    window.width -= cmax-gcols+1;
  }
  if (rmin<0)
  {
    window.y -= rmin;
    window.height += rmin;
  }
  if (cmin<0)
  {
    window.x -= cmin;
    window.width += cmin;
  }

  cv::Mat global_map_roi = global_map.map(cv::Rect(cmin+window.x, rmin+window.y, window.width, window.height));
  cv::Mat res_roi = res(window);
  global_map_roi.copyTo(res_roi);

  return res;
}

cv::Mat
MapManager::getMapWindow()
{
  double xMinLoc, yMinLoc;
  getWorldCoordinatesAbs(0, 0, xMinLoc, yMinLoc);

  int rMinGlob, cMinGlob, rMaxGlob, cMaxGlob;
  getGlobalMapCoordinates(xMinLoc, yMinLoc, rMinGlob, cMinGlob);
  rMaxGlob = rMinGlob + 2*hsize;
  cMaxGlob = cMinGlob + 2*hsize;

  /* copy all matrix from (rMinGlob, cMinGlob) to (rMaxGlob, cMaxGlob) */
  // -> same size and alignment of local map
  cv::Mat global_view = getGlobalView(rMinGlob, cMinGlob, rMaxGlob, cMaxGlob);

  cv::Mat result;
  cvtColor(global_view, result, cv::COLOR_GRAY2BGR);

  return result;
}

bool 
MapManager::getMapCoordinates(double x, double y, int & r, int & c)
{
  //c = x/resolution + sgn(x)*0.5 + hsize;
  //r = y/resolution + sgn(y)*0.5 + hsize;
  c = x/resolution + hsize;
  r = y/resolution + hsize;
  return c>=0 && c<=2*hsize && r>=0 && r<=2*hsize;
}

bool 
MapManager::getMapCoordinatesAbs(double x, double y, int & r, int & c)
{
  x -= origin.x;
  y -= origin.y;
//  c = x/resolution + sgn(x)*0.5 + hsize;
//  r = y/resolution + sgn(y)*0.5 + hsize;
  c = x/resolution + hsize;
  r = y/resolution + hsize;
  return c>=0 && c<=2*hsize && r>=0 && r<=2*hsize;
}

void
MapManager::getGlobalMapCoordinates(double x, double y, int & r, int & c)
{
  x -= global_map.pose.x;
  y -= global_map.pose.y;
  c = x/resolution; // + sgn(x)*0.5;
  r = y/resolution; // + sgn(y)*0.5;
}

bool 
MapManager::getWorldCoordinates(int r, int c, double & x, double & y)
{
  x = (c - hsize) * resolution;
  y = (r - hsize) * resolution;
  return c>=0 && c<=2*hsize && r>=0 && r<=2*hsize;
}

bool 
MapManager::getWorldCoordinatesAbs(int r, int c, double & x, double & y)
{
  x = (c - hsize) * resolution + origin.x;
  y = (r - hsize) * resolution + origin.y;
  return c>=0 && c<=2*hsize && r>=0 && r<=2*hsize;
}


