#include "pathFitter.hpp"
#include "smooth.hpp"

#include "utils.hpp"
#include "hardwareglobalinterface.hpp"
#include <ClothoidAsyPlot.hh>

#include <type_traits>
#include <algorithm>
#include <vector>

using G2lib::AsyPlot;
using std::vector;

// inline static
// int findAtS(double s, int & idx, std::vector<double> const & s0);



PathFitter::PathFitter() {
}
    
void PathFitter::push_back(double x, double y) {
  
  std::unique_lock<std::mutex> lock(ptsMtx);
  
  if (pts.empty()) { // init with current robot pose
    RobotStatus::LocalizationData locData;
    HardwareGlobalInterface::getInstance()->getLocalizationData(locData);
    Point pt(t0++, locData.x, locData.y);
    pts.push_back(pt);
  }
  else {
    double dst = std::hypot(x-pts.back().x, y-pts.back().y);
    if (dst<0.2) return;
  }
  
  Point pt(t0++, x, y);
  pts.push_back(pt);

  if (pts.size()<4) return; // TODO: remove

  int n = pts.size();
  VectorXd ts(n), xs(n), ys(n);
  for (int i=0; i<n; ++i) {
    ts(i) = pts[i].t;
    xs(i) = pts[i].x;
    ys(i) = pts[i].y;
  }

  lock.unlock();
  
  // LOESS FITTING
  //double ratio = std::max(0.25, 4./n);
  int span = std::min(n-1, 20); //(int)std::ceil(ratio*n));
  span = span%2?span:span+1;

  VectorXd xF, yF;
  if (n<4) {
    xF = xs;
    yF = ys;
  }
  else {
    xF = unifloess(xs, span, true);
    yF = unifloess(ys, span, true);
  }
  
  fit.resize(n);
  for (int i=0; i<n; ++i) {
    fit[i].t = ts(i);
    fit[i].x = xF(i);
    fit[i].y = yF(i);
  }

  // UNFIFORM SAMPLING 
  G2lib::PolyLine sp;
  
  vector<double> s0;
  s0.reserve(fit.size());
  s0.push_back(0.);
  
  sp.init(fit[0].x, fit[0].y);
  for (int i=1; i<fit.size(); ++i) {
    sp.push_back(fit[i].x, fit[i].y);
    s0.push_back(s0[i-1]+sp.getSegment(i-1).length());
  }


  int ncuts = ceil(sp.length()/0.4);
  double dL = sp.length()/ncuts;
  vector<double> xv, yv, thetav;
  
  vector<int> stN;
  for (int i=0; i<=ncuts; ++i) {
    double s = i*dL;

    double xc, yc, thetac, kappac;
    sp.evaluate(s, thetac, kappac, xc, yc);
    xv.push_back(xc);
    yv.push_back(yc);
    thetav.push_back(thetac);

    int idx = sp.findAtS(s);
    int t0 = fit[idx].t;
    stN.push_back(t0);
  }
  
  auto tguess = thetav;


  // G2lib::ClothoidList cl;
  // for (int i=0; i<sp.numSegment(); ++i) {
  //   cl.push_back(sp.getSegment(i));
  // }
  

  G2lib::ClothoidList cl;
  for (int i=1; i<xv.size(); ++i) {
    if (i==1) {
      G2lib::ClothoidCurve cc;
      cc.build_G1(xv[i-1], yv[i-1], tguess[i-1], xv[i], yv[i], tguess[i]);
      cl.push_back(cc);  
    }
    else {
      cl.push_back_G1(xv[i], yv[i], tguess[i]);
    }
  }

  std::cerr << cl.length() << std::endl;
  
  
  setPath(cl, stN);
}
       
void PathFitter::plot() {
  AsyPlot figure("plot.asy", false);

  int n = pts.size();
  ::plot(figure, 0, n, [&](int i, double& x, double& y) { 
    x = pts[i].x;
    y = pts[i].y;
  }, "red");
  ::plot(figure, 0, n, [&](int i, double& x, double& y) { 
    x = fit[i].x;
    y = fit[i].y;
  }, "green+dashed");
}

std::vector<Point> PathFitter::fittedPts() {
  return fit;
}

void PathFitter::setPath(G2lib::ClothoidList const & path, vector<int> const & st) {
  std::unique_lock<std::mutex> lock(pathMtx);
  this->path = path;
  this->st = st;
}

std::pair<G2lib::ClothoidList, std::vector<int>> PathFitter::getPath() {
  std::unique_lock<std::mutex> lock(pathMtx);
  return { path, st };
}

void PathFitter::trim(int t0) {
  std::unique_lock<std::mutex> lock(ptsMtx);
  auto iter = std::lower_bound(pts.begin(), pts.end(), t0,
    [](const Point & lhs, int rhs) {
      return lhs.t < rhs;
    }
  );

  pts.erase(pts.begin(), iter);
}



// #define ERROR(MSG)                     \
// {                                      \
//   std::ostringstream ost;              \
//   ost << "On line: " << __LINE__       \
//       << " file: " << __FILE__         \
//       << '\n' << MSG << '\n';          \
//   throw std::runtime_error(ost.str()); \
// }

//#define ASSERT(COND, MSG) if ( !(COND) ) ERROR(MSG)

// inline static
// int findAtS(double s, int & idx, std::vector<double> const & s0);
// {
//   int ns = int(s0.size()-1);
//   ASSERT(
//     idx >= 0 && idx < ns,
//     "findAtS( s=" << s << ", idx=" << idx << ",... ) bad index"
//   )
  
//   using const_s0_it = std::vector<double>::const_iterator const;
//   const_s0_it itL = std::next(s0.cbegin(), idx);
//     if ( s < *itL ) {
//       if ( s > s0.front() ) {
//         const_s0_it itB = s0.cbegin();
//         idx = int(std::distance(itB, lower_bound( itB, itL, s )));
//       } else {
//         idx = 0;
//       }
//     } else if ( s > *std::next(itL, 1) ) {
//       if ( s < s0.back() ) {
//         const_s0_it itE = s0.cend(); // past to the last
//         idx += int(std::distance(itL, lower_bound( itL, itE, s )));
//       } else {
//         idx = ns-1;
//       }
//     } else {
//       return idx; // vale intervallo precedente
//     }
//     if ( s0[size_t(idx)] > s ) --idx; // aggiustamento caso di bordo
//     ASSERT(
//       idx >= 0 && idx < ns,
//       "findAtS( s=" << s << ", idx=" << idx <<
//       ",... ) range [" << s0.front() << ", " << s0.back() << "]"
//     )
//     return idx;
// }
