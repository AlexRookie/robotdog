#pragma once

#include <ClothoidAsyPlot.hh>
#include <ClothoidList.hh>
#include <Clothoid.hh>


template<typename Iterator, typename Functor>
inline void plot(const G2lib::AsyPlot & figure, Iterator begin, Iterator end, Functor functor, std::string pen  = "black") {
  static_assert(std::is_invocable<Functor, Iterator, double&, double&>::value, "Functor not compatible with iterator");
  if (end==begin) return;
  double x0, y0;
  functor(begin, x0, y0); 
  ++begin;
  for (Iterator it=begin; it!=end; ++it) {
    double x1, y1;
    functor(it, x1, y1); 
    figure.drawLine(x0, y0, x1, y1, pen);
    x0 = x1;
    y0 = y1;
  }
}

inline void plotCircles(const G2lib::AsyPlot & figure, const std::vector<double> & x, const std::vector<double> & y, double radius, std::string pen  = "black") {
  for (int i=0; i<x.size(); ++i) {
    double xc = x[i]+radius, yc = y[i], tc = M_PI/2.;
    double kc = 1./radius, dkc = 0., Lc = 2*M_PI*radius;
    G2lib::ClothoidCurve cc(xc, yc, tc, kc, dkc, Lc);
    figure.drawClothoid(cc, pen);
  }
}

inline void plotClothoidList(const G2lib::AsyPlot & figure, const G2lib::ClothoidList & cl, std::string pen = "black") {
  for (int i=0; i<cl.numSegment(); ++i) {
    figure.drawClothoid(cl.get(i), pen);
  }
}

