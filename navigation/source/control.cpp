#include "control.hpp"

using G2lib::ClothoidList;

/// @brief Gives the sign of a variable
template <typename T> 
static inline int sgn(T val){
 return (T(0)<val) - (val < T(0));
}

static inline double diffangle(double beta, double alpha){
    //set alpha, beta between 0 and 2*pi
    while(fabs(beta)>=2.0*M_PI){
        beta = beta - 2.0*M_PI*sgn(beta);
    }
    if(beta<0.0){
        beta = beta + 2.0*M_PI;
    }
    while(fabs(alpha)>2.0*M_PI){
        alpha = alpha - 2.0*M_PI*sgn(alpha);
    }
    if(alpha<0.0){
        alpha = alpha + 2.0*M_PI;
    }
    double difference = beta - alpha;
    if(difference>M_PI){
        difference = difference - 2.0*M_PI;
    }
    if(difference<-M_PI){
        difference = difference + 2.0*M_PI;
    }
    return difference;
}


struct DynamicFrenetPoint{
    double x;
    double y;
    double l_x;   ///< The x-distance of the agent from the path.
    double l_y;   ///< The y-distance of the agent from the path.
    double theta_tilde;  ///< The yaw error
    double s;   ///< The curvilinear abscissa of the Frenet Point.
    double c;   ///< The curvature of the Frenet Point.
    double dc; ///< The derivative of the curvature of the preview point

    /// @brief Constructor.
    ///
    /// This is the constructor of the structure.
    DynamicFrenetPoint(): x(0.0f), y(0.0f), l_x(0.0f), l_y(0.0f), theta_tilde(0.0f), s(0.0f), c(0.0f), dc(0.0f) { }

    /// @brief Constructor.
    ///
    /// This is the copy constructor of the structure.
    DynamicFrenetPoint(const DynamicFrenetPoint &in) :
        x(in.x),
        y(in.y),
        l_x(in.l_x),
        l_y(in.l_y),
        theta_tilde(in.theta_tilde),
        s(in.s),
        c(in.c),
        dc(in.dc) { }

};


static void computeDFP(double rx, double ry, double rtheta, double ps, double px, double py, double ptheta, double pc, double pdc, DynamicFrenetPoint & dfp) {
  dfp.x = px;
  dfp.y = py;
  dfp.l_x = std::cos(ptheta)*(rx - px) + std::sin(ptheta)*(ry - py);
  dfp.l_y = std::cos(ptheta)*(ry - py) - std::sin(ptheta)*(rx - px);

  dfp.theta_tilde = diffangle(rtheta, ptheta);
  dfp.s = ps;
  dfp.c = pc;
  dfp.dc = pdc;
}

void closestPathPoint(ClothoidList const & path, 
                      double rx, double ry, double rtheta,
                      double & ps, 
                      double & px, double & py, double & ptheta, 
                      double & pc, double & pdc) {
  double xs, ys, s, t, dst;
  int i_curve;
  path.closestPointInRange_ISO(rx, ry, 0, path.numSegment()-1, xs, ys, s, t, dst, i_curve);
  s = s>=0 ? s : 0;
  s = s<=path.length() ? s : path.length();
  ps = s;
  px = xs;
  py = ys;
  ptheta = path.theta(s);
  pc = path.kappa(s);
  pdc = path.kappa_D(s);
}


double computeControl(double DT, ClothoidList const & path, double & vCmd, double & omegaCmd) {
  // parameters
  // LIMITS
  const double A_MAX = 0.5, V_MAX = 0.6, O_MAX = M_PI, A_LAT_MAX = 0.3;
  // OMEGA
  const double kappa_p = 1.0, kappa_x_p = 1.0, approach_p = 1.0;
  // VELOCITY
  const double distance_p = 0.5;
  const double kp_v = 1.0;

  if (path.numSegment() == 0) {
    vCmd = 0.0;
    omegaCmd = 0.0;
    return -1;
  }

  RobotStatus::LocalizationData locData;
  HardwareGlobalInterface::getInstance()->getLocalizationData(locData);
  double rx = locData.x, ry = locData.y, rtheta = locData.theta; // robot position

  RobotStatus::OdometryData odomData;
  HardwareGlobalInterface::getInstance()->getOdomData(odomData);
  double rv = odomData.v, romega = odomData.omega; // robot velocity

  // determine closest point [WINDOW??]
  double ps, px, py, ptheta, pc, pdc; // closest point x, y, theta
  closestPathPoint(path, rx, ry, rtheta, ps, px, py, ptheta, pc, pdc); //[window aroun s0/from s0 onwards??]

  // compute DFP parameters
  DynamicFrenetPoint dfp;
  computeDFP(rx, ry, rtheta, ps, px, py, ptheta, pc, pdc, dfp);
    
  // compute OMEGA
  double xi_dot = std::cos(dfp.theta_tilde) + kappa_x_p*dfp.l_x;
  double s_dot = rv*xi_dot;
  double l_x_dot = -s_dot*(1-dfp.c*dfp.l_y)+rv*std::cos(dfp.theta_tilde);
  double theta_tilde_dot = romega - dfp.c*s_dot;
  double xi_ddot = -std::sin(dfp.theta_tilde)*theta_tilde_dot + kappa_x_p*2.0*dfp.l_x*l_x_dot;

  if(dfp.s<0.01 && s_dot<0.0){
      s_dot   = 0.0;
      xi_dot  = 0.0;
      xi_ddot = 0.0;
  }

  double delta = -M_PI_2 * std::tanh(dfp.l_y * approach_p);

  double diff_delta_diff_ly = -M_PI/2.0 * approach_p * (1.0 - std::pow(std::tanh(dfp.l_y * approach_p), 2.0));
  double gamma = dfp.c*xi_dot + (-dfp.c*xi_dot*dfp.l_x + std::sin(dfp.theta_tilde))*diff_delta_diff_ly;

  double rho = gamma - kappa_p*diffangle(dfp.theta_tilde, delta);
  double w_d_unicycle = rv*rho;

  // compute V
  double delta_s = path.length() - dfp.s;
  double error_s = delta_s - distance_p;
  static double v_des_old = 0., error_s_old = 0.;

  //std::cout << "e_s: " << error_s << " e_s_old: " << error_s_old << std::endl; 

  double acc_dt = kp_v*error_s;

  acc_dt = std::min(acc_dt, A_MAX*DT);
  acc_dt = std::max(acc_dt, -A_MAX*DT);
  //std::cout << "a_dt: " << acc_dt << std::endl;
  
  double v_des = v_des_old + acc_dt;
  v_des = std::max(0., std::min(v_des, V_MAX));

  double v_max = V_MAX;

  G2lib::ClothoidList clt = path;
  clt.trim(dfp.s, std::min(dfp.s+0.7, clt.length()));
  double kMaxAbs = 0.;
  for (int i=0; i<clt.numSegment(); ++i) {
    double kMin, kMax;
    clt.get(i).curvatureMinMax(kMin, kMax);
    kMaxAbs = std::max(kMaxAbs, std::max(std::abs(kMin), std::abs(kMax)));
  }
  double brakingCoeff = 1. - 0.4*std::tanh(3*kMaxAbs);
  v_max *= brakingCoeff; 
  
  if (v_des>v_max) {
    double a_req = (v_des-v_max)/DT;
    if (a_req<=A_MAX) {
      v_des = v_max;
    }
    else {
      v_des = v_des - A_MAX*DT;
    }
  }

  double sigma = 1;
  if (w_d_unicycle>O_MAX) {
    sigma = w_d_unicycle/O_MAX;
    w_d_unicycle = O_MAX;
  }
  else if (w_d_unicycle<-O_MAX) {
    sigma = -w_d_unicycle/O_MAX;
    w_d_unicycle = -O_MAX;
  }
  v_des = v_des/sigma;

  double a_lat = v_des*std::abs(w_d_unicycle);
  double r = std::sqrt(A_LAT_MAX/a_lat);
  if (a_lat>A_LAT_MAX) {
    v_des = v_des * r;
    w_d_unicycle = w_d_unicycle * r;
  }
  
  //std::cout << "v_des: " << v_des << std::endl;

  error_s_old = error_s;
  v_des_old = v_des;

  vCmd = v_des;
  omegaCmd = w_d_unicycle;  
  
  return dfp.s;
}
