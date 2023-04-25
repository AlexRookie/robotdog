#pragma once

#include <ClothoidList.hh>

#include <vector>

struct Point { 
  int t;
  double x, y;

  Point(int t, double x, double y): t(t), x(x), y(y) {}
  Point(): Point(0, 0., 0.) {}
};

class PathFitter {
private:
  int t0 = 0;
  std::vector<Point> pts;
  std::vector<Point> fit;
  std::vector<int>   st; // max t up to the beginning of the i-th segment

  std::mutex pathMtx, ptsMtx;
  G2lib::ClothoidList path;

  void setPath(G2lib::ClothoidList const & path, std::vector<int> const & st);

public:
  PathFitter();
  void push_back(double x, double y);
  void plot();
  std::vector<Point> fittedPts();
  std::pair<G2lib::ClothoidList, std::vector<int>> getPath();
  void trim(int t0);

};
