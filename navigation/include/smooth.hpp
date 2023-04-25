#pragma once

#include <Eigen/Dense>
#include <Eigen/QR>

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

VectorXd filter(VectorXd & B, VectorXd & A, VectorXd const & X, VectorXd & SI);

VectorXd filter(VectorXd & B, VectorXd & A, VectorXd const & X);

VectorXd unifloess(VectorXd const & y, int span, bool useLoess);

