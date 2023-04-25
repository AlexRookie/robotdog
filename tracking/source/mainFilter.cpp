//============================================================================/
// Main filter
//
// Alessandro Antonucci @AlexRookie
// Paolo Bevilacqua
// University of Trento
//============================================================================/

#include <stdio.h>
#include <stdlib.h> // exit(0);>
#include <iostream>
#include <fstream>

#include <cmath>
#include <vector>

#include <random>

#include <chrono>
#include <ctime>
#include <string>

#include <signal.h>
#include <unistd.h>
#include <getopt.h>

#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp> // OpenCV
#include "zmq/Publisher.hh"
#include "zmq/Subscriber.hh"
#include "json.hpp"
#include "kalman.hpp"

using nlohmann::json;

std::string pubAddress = "tcp://*:3220"; 

std::string subAddressHuman = "tcp://127.0.0.1:3218"; //parser.get<string>("pubAddress");
std::string subAddressLoc   = "tcp://10.196.80.135:9207"; 
std::string subAddressOdom  = "tcp://10.196.80.135:9307"; 

std::string subTopicHuman = "HUM_POS"; 
std::string subTopicLoc   = "POS";
std::string subTopicOdom  = "ODOM"; 

#define DELTAT       50 // ms
#define STATES       5
#define OUTPUTS      2
#define MAX_FAILURES 5

//#define DEBUG

struct EulerAngles {
  double roll, pitch, yaw;
};

struct Quaternion {
  double x, y, z, w;
};

EulerAngles ToEulerAngles(Quaternion const & q);

static inline void delay(int ms){
    while (ms>=1000){
        usleep(1000*1000);
        ms-=1000;
    };
    if (ms!=0)
        usleep(ms*1000);
}

bool ctrl_c_pressed;
void ctrlc(int) {
    ctrl_c_pressed = true;
}

long long int timeNow() {
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    using std::chrono::system_clock;
    return (long long int)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// (Non)-linear measurement matrix (m,n)
Matrix h = {{1, 0, 0, 0, 0},
            {0, 1, 0, 0, 0}};

// Jacobian matrix (m,n)
Matrix H = {{1, 0, 0, 0, 0},
            {0, 1, 0, 0, 0}};

// Linearized state transition matrix A (n,n)
template <class T>
std::vector<std::vector<T>> A_lin(std::vector<T> const &v, double const &dt) {
    std::vector<std::vector<T>> A(STATES, std::vector<T>(STATES, 0));
    // 1-st row
    A[0][0] = 1.;                          // diff(xn,x)
    A[0][1] = 0.;                          // diff(xn,y)
    A[0][2] = -dt * v[3] * std::sin(v[2]); // diff(xn,theta)
    A[0][3] = dt * std::cos(v[2]);         // diff(xn,v)
    A[0][4] = 0.;                          // diff(xn,omega)
    // 2-nd row
    A[1][0] = 0.;                         // diff(yn,x)
    A[1][1] = 1.;                         // diff(yn,y)
    A[1][2] = dt * v[3] * std::cos(v[2]); // diff(yn,theta)
    A[1][3] = dt * std::sin(v[2]);        // diff(yn,v)
    A[1][4] = 0.;                         // diff(yn,omega)
    // 3-rd row
    A[2][0] = 0.;       // diff(thetan,x)
    A[2][1] = 0.;       // diff(thetan,y)
    A[2][2] = 1.;       // diff(thetan,theta)
    A[2][3] = 0.;       // diff(thetan,v)
    A[2][4] = dt;       // diff(thetan,omega)
    // 4-th row
    A[3][0] = 0.;       // diff(vn,x)
    A[3][1] = 0.;       // diff(vn,y)
    A[3][2] = 0.;       // diff(vn,theta)
    A[3][3] = 1.;       // diff(vn,v)
    A[3][4] = 0.;       // diff(vn,omega)
    // 5-th rowMeas
    A[4][0] = 0.;       // diff(vn,x)
    A[4][1] = 0.;       // diff(vn,y)
    A[4][2] = 0.;       // diff(vn,theta)
    A[4][3] = 0.;       // diff(vn,v)
    A[4][4] = 1.;       // diff(omegan,omega)
    return A;
}

// Linearized process noise gain matrix G (n,m)
template <class T>
std::vector<std::vector<T>> G_lin(std::vector<T> const &v, double const &dt) {
    std::vector<std::vector<T>> G(STATES, std::vector<T>(OUTPUTS, 0));
    // 1-st row
    G[0][0] = 0.; // diff(xn,acc)
    G[0][1] = 0.; // diff(xn,d_omega)
    // 2-nd row
    G[1][0] = 0.; // diff(yn,acc)
    G[1][1] = 0.; // diff(yn,d_omega)
    // 3-rd row
    G[2][0] = 0.; // diff(thetan,acc)
    G[2][1] = 0.; // diff(thetan,d_omega)    
    // 4-th row
    G[3][0] = dt; // diff(vn,acc)
    G[3][1] = 0.; // diff(vn,d_omega)  
    // 5-th row
    G[4][0] = 0.; // diff(omegan,acc)
    G[4][1] = dt; // diff(omegan,d_omega)
    return G;
}

// Process noise covariance matrix Q (n,n) [????]
double stdQvel = 0.5;
Matrix Q1 = {{std::pow(stdQvel,2), 0},
            {0, std::pow(stdQvel,2)}};
double stdQacc     = 0.3;
double stdQd_omega = 0.3;
Matrix Q2 = {{std::pow(stdQacc,2), 0},
            {0, std::pow(stdQd_omega,2)}};

// Measurement noise covariance matrix R (m,m)
double stdR = 0.5;
Matrix R = {{std::pow(stdR,2), 0},
            {0, std::pow(stdR,2)}};

// System input (control)
Matrix B = {{0.},{0.},{0.},{0.},{0.}}; // (n,u)
Vector u = {0.};


int main(int argc, char *argv[]) {
    signal(SIGINT, ctrlc);

    std::cout << "[Initialization...]" << std::endl;
    
    ////////////////////////////////////////////////////////////////////////////
    // PUBLISHER FILTER POSE
    ////////////////////////////////////////////////////////////////////////////

    Publisher pub(pubAddress);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO HUMAN POSE
    ////////////////////////////////////////////////////////////////////////////
    
    struct TimedPose {
        double x = 0, y = 0;
        bool valid = false;
        long long int timestamp = 0;
        std::mutex mtx;
    };

    Subscriber sub2D;
    TimedPose trackedPose;
    sub2D.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        try {
            json jobj = json::parse(std::string(buf, size));
            long long int timestamp = jobj.at("ts");
            bool valid = jobj.at("valid");
            double x = jobj.at("x");
            double y = jobj.at("y");
            {
                std::unique_lock<std::mutex> lck(trackedPose.mtx);
                trackedPose.x = x;
                trackedPose.y = y;
                trackedPose.valid = valid;
                trackedPose.timestamp = timestamp;
            }
        }
        catch(std::exception &e) {
            std::cerr << "Error parsing human tracking data: " << e.what() << std::endl;
        }
    });
    sub2D.start(subAddressHuman, subTopicHuman);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO LOC
    ////////////////////////////////////////////////////////////////////////////

    struct Pose {
        double x, y, theta;
        long long int ts;
        std::mutex mtx;
    };

    Subscriber subLoc;
    Pose locData;
    subLoc.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        nlohmann::json j;
        
        try {
            j = nlohmann::json::parse(std::string(buf, size));
            double x, y, theta;
            long long int ts;
            x = j.at("loc_data").at("x");
            y = j.at("loc_data").at("y");
            theta = j.at("loc_data").at("theta");
            ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            {
                std::unique_lock<std::mutex> lock(locData.mtx);
                locData.x = x;
                locData.y = y;
                locData.theta = theta;
                locData.ts = ts;
            }
        }
        catch(std::exception &e)  {
            std::cerr << "\"" << std::string(buf, size) << "\"" << std::endl;
            std::cerr << "error parsing loc data: " << e.what() << std::endl;
        }
    });
    subLoc.start(subAddressLoc, subTopicLoc);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO ODOM
    ////////////////////////////////////////////////////////////////////////////

    struct Odometry {
        double x, y, theta;
        double vx, vy, vtheta;
        long long int ts;
        std::mutex mtx;
    };

    Subscriber subOdom;
    Odometry odomData;
    subOdom.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        nlohmann::json j;

        try {
            j = nlohmann::json::parse(std::string(buf, size));   
            double x, y, theta;
            double vx, vy, vtheta;
            long long int ts;
            ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            
            x = j["pose"]["pose"]["position"]["x"]; 
            y = j["pose"]["pose"]["position"]["y"];

            Quaternion q;
            q.x = j["pose"]["pose"]["orientation"]["x"];
            q.y = j["pose"]["pose"]["orientation"]["y"];
            q.z = j["pose"]["pose"]["orientation"]["z"];
            q.w = j["pose"]["pose"]["orientation"]["w"];
            EulerAngles angles = ToEulerAngles(q);
            theta = angles.yaw;
                    
            vx = j["twist"]["twist"]["linear"]["x"];
            vy = j["twist"]["twist"]["linear"]["y"];
            vtheta = j["twist"]["twist"]["angular"]["z"];
            
            {
                std::unique_lock<std::mutex> lock(odomData.mtx);
                odomData.x = x;
                odomData.y = y;
                odomData.theta = theta;
                odomData.vx = vx;
                odomData.vy = vy;
                odomData.vtheta = vtheta;
                odomData.ts = ts;
            }

        }
        catch(std::exception &e) {
            std::cerr << "Error parsing odom data: " << e.what() << std::endl;
        }
    });
    subOdom.start(subAddressOdom, subTopicOdom);

    // Open log file
    // std::ofstream logfile;
    // logfile.open("./filter_data.txt");
    
    // Initialize states
    Matrix X1, X2;

    // Initialize state covariance matrix
    Matrix P = {{1e5, 0, 0, 0, 0},
                {0, 1e5, 0, 0, 0},
                {0, 0, 1e5, 0, 0},
                {0, 0, 0, 1e5, 0},
                {0, 0, 0, 0, 1e5}};
    Matrix P1 = P;
    Matrix P2 = P;

    // Define random generator with Gaussian distribution
    std::mt19937 generator(std::random_device{}()); // Mersenne twister PRNG

    std::normal_distribution<double> randn_vel(0.0, std::sqrt(Q1[0][0]));
    std::normal_distribution<double> randn_acc(0.0, std::sqrt(Q2[0][0]));
    std::normal_distribution<double> randn_domega(0.0, std::sqrt(Q2[1][1]));

    bool re_start = true;
    int failcount = 0;
    long long int tic = timeNow();

    std::cout << "[Looping...]" << std::endl;

    while (1) {
        // Exit
        if (ctrl_c_pressed) {
            break;
        }

        long long int toc = timeNow();
        double dt = (double)((toc-tic)/1000.);
        tic = toc;
        //std::cout << "dt: " << dt << std::endl;

        #ifdef DEBUG
            // Plot map grid
            int l = 800;
            cv::Mat plot(l ,l, CV_8UC3, cv::Scalar(255,255,255));
            for (int i=50; i<=l; i+=50) {
                cv::line(plot, cv::Point(i, 0), cv::Point(i, l), cv::Scalar(128,128,128), 1);
                cv::line(plot, cv::Point(0, i), cv::Point(l, i), cv::Scalar(128,128,128), 1);
            }
        #endif

        TimedPose pose2Drel;
        {
            std::unique_lock<std::mutex> lck(trackedPose.mtx);
            pose2Drel.x = trackedPose.x;
            pose2Drel.y = trackedPose.y;
            pose2Drel.valid = trackedPose.valid;
            pose2Drel.timestamp = trackedPose.timestamp;
        }
        
        if (pose2Drel.valid == false) {
            failcount++;
            if (failcount >= MAX_FAILURES) {
                re_start = true;
            }
        }
        else {
            failcount = 0;
        }

        // Get measurement
        Pose locDataC;
        {
           std::unique_lock<std::mutex> lock(locData.mtx);
           locDataC.x = locData.x;
           locDataC.y = locData.y;
           locDataC.theta = locData.theta;
           locDataC.ts = locData.ts;
       }

        bool validLoc = true; //std::abs(locDataC.ts-timeNow()) < 100;
        if (!validLoc) std::cerr << "No localization received, or too late!!!" << std::endl;

        double pose2Dabs_x = 0, pose2Dabs_y = 0;
        Vector meas;
        if (validLoc && pose2Drel.valid) {
            double cost = std::cos(locDataC.theta);
            double sint = std::sin(locDataC.theta);
            pose2Dabs_x = cost*pose2Drel.x - sint*pose2Drel.y + locDataC.x;
            pose2Dabs_y = sint*pose2Drel.x + cost*pose2Drel.y + locDataC.y;
            meas = {pose2Dabs_x, pose2Dabs_y};
        }

        if (re_start == true) {
            if (pose2Drel.valid) {
                // Initialize states
                X1.clear();
                X2.clear();
                X1.push_back({pose2Dabs_x, pose2Dabs_y, 0., 0., 0.});
                X2.push_back({pose2Dabs_x, pose2Dabs_y, 0., 0., 0.});
                P1 = P;
                P2 = P;
                re_start = false;
                failcount = 0;
                std::cout << "ReStart" << std::endl;
            }
            else {
                delay(20);
                continue;
            }
        }
        
        // Generate noise
        Vector lin_acc = {randn_vel(generator), randn_vel(generator)};
        double acc = randn_acc(generator);
        double d_omega = randn_domega(generator);
        
        //---- KF model ----
        // x = Ax + Bu + Gv
        // z = Hx + e

        // Linear state transition matrix A (n,n)
        Matrix A1 = {{1, 0, dt, 0,  0},
                     {0, 1, 0,  dt, 0},
                     {0, 0, 1,  0,  0},
                     {0, 0, 0,  1,  0},
                     {0, 0, 0,  0,  0}};

        // Linear process noise gain matrix G (n,m)
        Matrix G1 = {{std::pow(dt,2)/2, 0 },
                     {0,  std::pow(dt,2)/2},
                     {dt, 0 },
                     {0,  dt},
                     {0,  0 }};

        // Linear state estimate
        Vector x1_ = vector_sum(vector_sum(matrix_vector(A1, X1.back()), matrix_vector(B, u)), matrix_vector(G1, lin_acc));
        // State estimate covariance
        Matrix P1_ = matrix_sum(matrix_matrix(matrix_matrix(A1, P1), transpose(A1)), matrix_matrix(matrix_matrix(G1, Q1), transpose(G1)));

        if (validLoc && pose2Drel.valid) {
            // Innovation (when 'linear', h=H))
            Vector r1 = vector_diff(meas, matrix_vector(h, x1_));
            // Innovation covariance
            Matrix S1 = matrix_sum(matrix_matrix(matrix_matrix(H, P1_), transpose(H)), R);
            // filter gain
            Matrix K1 = matrix_matrix(matrix_matrix(P1_, transpose(H)), inverse(S1));
                
            // Updated state estimate
            X1.push_back( vector_sum(x1_, matrix_vector(K1, r1)) );
            // Updated state estimate covariance
            P1 = matrix_diff(P1_, matrix_matrix(matrix_matrix(K1, H), P1_));
        }
        else {
            X1.push_back(x1_);
            P1 = P1_;
        }

        //---- EKF model ----
        // x = f(x,u,v)
        // z = h(x,e)

        // Non-linear state transition
        Vector f(STATES);
        f[0] = X2.back()[0] + dt * X2.back()[3] * cos(X2.back()[2]);
        f[1] = X2.back()[1] + dt * X2.back()[3] * sin(X2.back()[2]);
        f[2] = X2.back()[2] + dt * X2.back()[4];
        f[3] = X2.back()[3] + dt * acc;
        f[4] = X2.back()[4] + dt * d_omega;
        // Linearized state transition matrix
        Matrix A2 = A_lin(X2.back(), dt);
        //  Linearized process noise gain matrix
        Matrix G2 = G_lin(X2.back(), dt);
        // Non-linear state estimate
        Vector x2_ = vector_sum(f, matrix_vector(B, u));
        // State covariance estimate   
        Matrix P2_ = matrix_sum(matrix_matrix(matrix_matrix(A2, P2), transpose(A2)), matrix_matrix(matrix_matrix(G2, Q2), transpose(G2))); 

        if (validLoc && pose2Drel.valid) {
            // Innovation (when 'linear', h=H))
            Vector r2 = vector_diff(meas, matrix_vector(h, x2_));
            // Innovation covariance
            Matrix S2 = matrix_sum(matrix_matrix(matrix_matrix(H, P2_), transpose(H)), R);
            // filter gain
            Matrix K2 = matrix_matrix(matrix_matrix(P2_, transpose(H)), inverse(S2));
                
            // Updated state estimate
            X2.push_back( vector_sum(x2_, matrix_vector(K2, r2)) );
            // Updated state estimate covariance
            P2 = matrix_diff(P2_, matrix_matrix(matrix_matrix(K2, H), P2_));
        }
        else {
            X2.push_back(x2_);
            P2 = P2_;
        }

        long long int currTs = timeNow();
        
        // Publish json
        json jobj;
        jobj["ts"] = currTs;
        jobj["meas_x_rel"] = pose2Drel.x;
        jobj["meas_y_rel"] = pose2Drel.y;
        jobj["meas_x_abs"] = pose2Dabs_x;
        jobj["meas_y_abs"] = pose2Dabs_y;
        jobj["meas_valid"] = pose2Drel.valid;
        jobj["loc_valid"] = validLoc;
        jobj["meas_ts"] = pose2Drel.timestamp;
        jobj["kf_x"] = X1.back()[0];
        jobj["kf_y"] = X1.back()[1];
        jobj["ekf_x"] = X2.back()[0];
        jobj["ekf_y"] = X2.back()[1];
        jobj["loc_x"] = locDataC.x;
        jobj["loc_y"] = locDataC.y;
        jobj["loc_theta"] = locDataC.theta;
        std::string jmsg = jobj.dump();
        pub.send("FIL_POS", jmsg.c_str(), jmsg.size());

        // Save to file
        //logfile << jmsg << std::endl;

        #ifdef DEBUG
            int px, py;

            // Plot tracked human
            if (pose2Drel.valid == true) {
                px = l/2 - (int)(pose2Drel.x*50);
                py = l/2 - (int)(pose2Drel.y*50);
                cv::rectangle(plot, cv::Point2f(px,py), cv::Point2f(px+5,py+5), cv::Scalar(255,0,0), -1);
            }

            // Plot filters
            px = l/2 - (int)(X1.back()[0]*50);
            py = l/2 - (int)(X1.back()[1]*50);
            cv::rectangle(plot, cv::Point2f(px,py), cv::Point2f(px+5,py+5), cv::Scalar(0,255,0), -1);
            px = l/2 - (int)(X2.back()[0]*50);
            py = l/2 - (int)(X2.back()[1]*50);
            cv::rectangle(plot, cv::Point2f(px,py), cv::Point2f(px+5,py+5), cv::Scalar(0,0,255), -1);
            
            cv::imshow("Filter", plot);
            int key = cv::waitKey(DELTAT);
        #else
            delay(DELTAT);
        #endif

        std::cout << "dt: " << dt << " fail: " << failcount
        << "Mx: " << pose2Drel.x << " My: " << pose2Drel.y
        << " X1: " << X1.back()[0] << " Y1: " << X1.back()[1]
        << " X2: " << X2.back()[0] << " Y2: " << X2.back()[1]
        << std::endl;
    }

    // Close log file
    //logfile.close();

    return 0;
}

EulerAngles ToEulerAngles(Quaternion const & q) {
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
