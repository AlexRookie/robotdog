#include "smooth.hpp"

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

template <typename T>
T diff(T const & in) {
  size_t n = in.size();
  if (n<2) return {};
  return in.tail(n-1)-in.head(n-1);
}

template <typename T>
Matrix<T, Dynamic, 1> linspace(T low, T step, T hi) {
  size_t sz = ((hi-low)/step)+1;
  return Matrix<T, Dynamic, 1>::LinSpaced(sz, low, low+step*(sz-1));
}

bool uniformx(VectorXd const & diffx, VectorXd const & x) {
  double xBeg = x(0);
  double xEnd = x(x.size()-1);
  double m = std::numeric_limits<double>::epsilon() * std::max(xBeg, xEnd);

  auto ddx = diff(diffx);
  for (int i=0; i<ddx.size(); ++i) {
    if (std::abs(ddx(i))>m) return false;
  }
  
  return true; 
}

double fix(double val) {
  if (val>0) return std::floor(val);
  else return std::ceil(val);
}

VectorXd filter(VectorXd & B, VectorXd & A, VectorXd const & X, VectorXd & SI) {
  //  Y = FILTER(B,A,X) filters the data in vector X with the
  //  filter described by vectors A and B to create the filtered
  //  data Y.  The filter is a "Direct Form II Transposed"
  //  implementation of the standard difference equation:
  // 
  //   a(1)*y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb)
  //                         - a(2)*y(n-1) - ... - a(na+1)*y(n-na)

  VectorXd Y;

  size_t a_len  = A.size();
  size_t b_len  = B.size();
  size_t x_len  = X.size();

  size_t si_len = SI.size();

  size_t ab_len = a_len > b_len ? a_len : b_len;
  

  B.conservativeResizeLike(VectorXd::Zero(ab_len));

  if (si_len != ab_len - 1)
  {
    throw std::runtime_error("filter: si must be a vector of length max (length (a), length (b)) - 1");
  }

  double norm = A(0);

  if (norm == 0.0)
  {
    throw std::runtime_error("filter: the first element of a must be non-zero");
  }

  Y = VectorXd::Zero(x_len);

  if (norm != 1.0) {
    B = B / norm;
  }

  if (a_len > 1)
  {
    A.conservativeResizeLike(VectorXd::Zero(ab_len));
    
    if (norm != 1.0) {
	    A = A / norm;
    }

    // T *py = y.fortran_vec ();
    // T *psi = si.fortran_vec ();

    // const T *pb = b.data ();
    // const T *pa = a.data ();
    // const T *px = x.data ();

    for (int i = 0; i < x_len; i++)
    {
	    Y(i) = SI(0) + B(0) * X(i);

	    if (si_len > 1)
	    {
	      for (int j = 0; j < si_len - 1; j++) {
		      SI(j) = SI(j+1) - A(j+1) * Y(i) + B(j+1) * X(i);
        }
	      SI(si_len-1) = B(si_len) * X(i) - A(si_len) * Y(i);
	    }
	    else {
	      SI(0) = B(si_len) * X(i) - A(si_len) * Y(i);
      }
	  }
  }
  else if (si_len > 0)
  {
    // T *py = y.fortran_vec ();
    // T *psi = si.fortran_vec ();

    // const T *pb = b.data ();
    // const T *px = x.data ();

    for (int i = 0; i < x_len; i++)
	  {
	    Y(i) = SI(0) + B(0) * X(i);

	    if (si_len > 1)
	    {
	      for (int j = 0; j < si_len - 1; j++) 
        {
		      SI(j) = SI(j+1) + B(j+1) * X(i);
        }
	      SI(si_len-1) = B(si_len) * X(i);
	    }
	    else 
      {
	      SI(0) = B(1) * X(i);
      }
	  }
  }
  else 
  {
    Y = B(0) * X;
  }

  return Y;
} 

VectorXd filter(VectorXd & B, VectorXd & A, VectorXd const & X)
{
  size_t a_len = A.size();
  size_t b_len = B.size();
  
  size_t si_len = (a_len > b_len ? a_len : b_len) - 1;
  VectorXd SI = VectorXd::Zero(si_len);

  return filter(B, A, X, SI);
}

VectorXd unifloess(VectorXd const & y, int span, bool useLoess) {
  // UNIFLOESS Apply loess on uniformly spaced X values

  // Omit points at the extremes, which have zero weight
  double halfw = (span-1)/2.;        // halfwidth of entire span
  int halfwI = round(halfw);
  VectorXd d = (linspace(1.-halfw, 1., halfw-1.)).cwiseAbs(); // distances to pts with nonzero weight
  double dmax = halfw;                    // max distance for tri-cubic weight
  

  // Set up weighted Vandermonde matrix using equally spaced X values
  auto x1 = linspace(2., 1., span-1.);
  x1.array() -= (halfw+1);
  auto weight = (1. - (d/dmax).array().cube()).array().pow(1.5); // tri-cubic weight
  
  MatrixXd v(x1.size(), 2);
  v.col(0) = MatrixXd::Ones(x1.size(), 1);
  v.col(1) = x1;

  if (useLoess) {
    v.conservativeResize(v.rows(), v.cols()+1);
    v.col(v.cols()-1) = x1.array().square();
  }

  MatrixXd tmp = weight.replicate(1, v.cols());
  MatrixXd V = v.cwiseProduct(tmp);


  // Do QR decomposition
  Eigen::HouseholderQR<MatrixXd> qr(V);
  MatrixXd Q = qr.householderQ() * MatrixXd::Identity(V.rows(), std::min(V.rows(), V.cols()));
    
  // The projection matrix is Q*Q'.  We want to project onto the middle
  // point, so we can take just one row of the first factor.  
  MatrixXd alpha = Q.row(halfwI-1)*Q.transpose();
  
  // This alpha defines the linear combination of the weighted y values that
  // yields the desired smooth values.  Incorporate the weights into the
  // coefficients of the linear combination, then apply filter.
  alpha = alpha.array().cwiseProduct(weight.transpose());
  
  VectorXd fA(1);
  fA(0) = 1;
  
  VectorXd alphaV = alpha.transpose();
  auto ys = filter(alphaV, fA, y);

  // We need to slide the values into the center of the array.
  int n = ys.size();

  ys.segment(halfwI, n-2*halfwI) = ys.segment(span-2, n-span+1);

  // Now we have taken care of everything except the end effects.  Loop over
  // the points where we don't have a complete span.  Now the Vandermonde
  // matrix has span-1 points, because only 1 has zero weight.
  x1 = linspace(1., 1., span-1.);
  v = MatrixXd(x1.size(), 2);
  v.col(0) = MatrixXd::Ones(x1.size(), 1);
  v.col(1) = x1;

  if (useLoess) {
    v.conservativeResize(v.rows(), v.cols()+1);
    v.col(v.cols()-1) = x1.array().square();
  }

  for (int j=1; j<=halfwI; ++j) {
    // Compute weights based on deviations from the jth point,
    // then compute weights and apply them as above. 
    VectorXd d = ((linspace(1., 1., span-1.)).array()-j).array().cwiseAbs();
    //std::cerr << d << std::endl << std::endl;

    auto weight = (1. - (d/(span-j)).array().cube()).array().pow(1.5);
    //std::cerr << weight << std::endl << std::endl;
    MatrixXd tmp = weight.replicate(1, v.cols());
    MatrixXd V = v.cwiseProduct(tmp);
    //std::cerr << V << std::endl << std::endl;

    Eigen::HouseholderQR<MatrixXd> qr(V);
    MatrixXd Q = qr.householderQ() * MatrixXd::Identity(V.rows(), std::min(V.rows(), V.cols()));
    //std::cerr << Q << std::endl << std::endl;

    MatrixXd alpha = Q.row(j-1)*Q.transpose();
    //std::cerr << alpha << std::endl << std::endl;

    alpha = alpha.array().cwiseProduct(weight.transpose());
    //std::cerr << alpha << std::endl << std::endl;

    ys(j-1) = (alpha * y.segment(0, span-1))(0);
    //std::cerr << ys << std::endl << std::endl;

    // These coefficients can be applied to the other end as well
    auto send = y.segment(n-span+1, span-1);
    
    //std::cerr << send.reverse() << std::endl << std::endl;

    ys(n-j) = (alpha * send.reverse())(0);
    //std::cerr << ys << std::endl << std::endl;
  }

  return ys;

// for j=1:halfw
//     % Compute weights based on deviations from the jth point,
//     % then compute weights and apply them as above.
//     d = abs((1:span-1) - j);
//     weight = (1 - (d/(span-j)).^3).^1.5;
//     V = v .* repmat(weight(:),1,size(v,2));
//     [Q,~] = qr(V,0);
//     alpha = Q(j,:)*Q';
//     alpha = alpha .* weight;
//     ys(j) = alpha * y(1:span-1);

//     % These coefficients can be applied to the other end as well
//     ys(end+1-j) = alpha * y(end:-1:end-span+2);
// end

}
