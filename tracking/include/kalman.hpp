// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef KALMAN_H
#define KALMAN_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

template <class T>
void print(std::vector<std::vector<T>> const &M) {
    for(int i=0; i < M.size(); i++) {
        for(int j=0; j < M[i].size(); j++) {
            std::cout << M[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

template <class T>
void print(std::vector<T> const &v) {
    for(int i=0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

template <class T>
std::vector<T> vector_sum(std::vector<T> const &a, std::vector<T> const &b) {
    std::vector<T> c;
    std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(c), std::plus<T>());
    return c;
}

template <class T>
std::vector<T> vector_diff(std::vector<T> const &a, std::vector<T> const &b) {
    std::vector<T> c;
    std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(c), std::minus<T>());
    return c;
}

template <class T>
double vector_multiply(std::vector<T> const &a, std::vector<T> const &b) {
    const int n = a.size();
    double c = 0;
    for (int i=0; i < n; i++) {
        c += a[i]*b[i];
    }
    return c;
}

template <class T>
std::vector<T> vector_dot_multiply(std::vector<T> const &a, std::vector<T> const &b) {
    const int n = a.size();
    std::vector<T> c(n, 0.);
    for (int i=0; i < n; i++) {
        c[i] = a[i]*b[i];
    }
    return c;
}

template <class T>
std::vector<std::vector<T>> transpose(std::vector<std::vector<T>> const &A) {
    const int n = A.size();    // A rows
    const int m = A[0].size(); // A cols
    std::vector<std::vector<T>> B(m, std::vector<T>(n, 0));
    
    for (int i=0; i < n; i++) {
        for (int j=0; j < m; j++) {
            B[j][i] = A[i][j];
        }
    }
    return B;
}

template <class T>
double determinant(std::vector<std::vector<T>> &A) { // TODO: modificare std::swap in modo che A sia const
    const double SMALL = 1.0E-30;
    int n = A.size();
    double det = 1;

    // Row operations for i = 0, ,,,, n - 2 (n-1 not needed)
    for (int i=0; i < n-1; i++ ) {
        // Partial pivot: find row r below with largest element in column i
        int r = i;
        double maxA = abs( A[i][i] );
        for ( int k =i+1; k < n; k++) {
            double val = abs( A[k][i] );
            if ( val > maxA ){
               r = k;
               maxA = val;
            }
        }
        if ( r != i ) {
            for (int j=i; j < n; j++ ) std::swap( A[i][j], A[r][j] );
            det = -det;
        }

        // Row operations to make upper-triangular
        double pivot = A[i][i];
        if ( abs( pivot ) < SMALL ) return 0.0; // singular matrix

        for (int r=i+1; r < n; r++)  {         // on lower rows
            double multiple = A[r][i] / pivot; // multiple of row i to clear element in ith column
            for (int j=i; j < n; j++) A[r][j] -= multiple * A[i][j];
        }
        det *= pivot; // determinant is product of diagonal
    }

    det *= A[n-1][n-1];
    return det;
}

template <class T>
std::vector<std::vector<T>> inverse(std::vector<std::vector<T>> const &A) {
    // Reference: https://martin-thoma.com/inverting-matrices/
    int n = A.size();
    std::vector<std::vector<T>> B(n, std::vector<T>(n));

    std::vector<T> line(2*n, 0);
    std::vector<std::vector<T>> Aline(n, line);

    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            Aline[i][j] = A[i][j];
        }
    }

    for (int i=0; i < n; i++) {
        Aline[i][n+i] = 1;
    }

    for (int i=0; i < n; i++) {
        // Search for maximum in this column
        double maxEl = abs(Aline[i][i]);
        int maxRow = i;
        for (int k=i+1; k < n; k++) {
            if (abs(Aline[k][i]) > maxEl) {
                maxEl = Aline[k][i];
                maxRow = k;
            }
        }

        // Swap maximum row with current row (column by column)
        for (int k=i; k < 2*n; k++) {
            double tmp = Aline[maxRow][k];
            Aline[maxRow][k] = Aline[i][k];
            Aline[i][k] = tmp;
        }

        // Make all rows below this one 0 in current column
        for (int k=i+1; k < n; k++) {
            double c = -Aline[k][i]/Aline[i][i];
            for (int j=i; j < 2*n; j++) {
                if (i==j) {
                    Aline[k][j] = 0;
                } else {
                    Aline[k][j] += c * Aline[i][j];
                }
            }
        }
    }

    // Solve equation Ax=b for an upper triangular matrix Aline
    for (int i=n-1; i >= 0; i--) {
        for (int k=n; k < 2*n; k++) {
            Aline[i][k] /= Aline[i][i];
        }

        for (int rowModify = i-1; rowModify >= 0; rowModify--) {
            for (int columModify = n; columModify< 2*n ; columModify++) {
                Aline[rowModify][columModify] -= Aline[i][columModify] * Aline[rowModify][i];
            }
        }
    }

    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            B[i][j] = Aline[i][n+j];
        }
    }
    return B;
}

template <class T>
std::vector<std::vector<T>> matrix_sum(std::vector<std::vector<T>> const &A, std::vector<std::vector<T>> const &B) {
    const int n = A.size();    // A rows
    const int m = A[0].size(); // A cols
    std::vector<std::vector<T>> C(n, std::vector<T>(m, 0));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

template <class T>
std::vector<std::vector<T>> matrix_diff(std::vector<std::vector<T>> const &A, std::vector<std::vector<T>> const &B) {
    const int n = A.size();    // A rows
    const int m = A[0].size(); // A cols
    std::vector<std::vector<T>> C(n, std::vector<T>(m, 0));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

template <class T>
std::vector<std::vector<T>> matrix_matrix(std::vector<std::vector<T>> const &A, std::vector<std::vector<T>> const &B) {
    const int n = A.size();    // A rows
    const int m = A[0].size(); // A cols
    const int p = B[0].size(); // B cols

    std::vector<std::vector<T>> C(n, std::vector<T>(p, 0));
    for (auto j=0; j < p; ++j) {
        for (auto k=0; k < m; ++k) {
            for (auto i=0; i < n; ++i) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

template <class T>
std::vector<T> matrix_vector(std::vector<std::vector<T>> const &M, std::vector<T> const &v) {
    const int n = M.size();    // M rows
    const int m = M[0].size(); // M cols
    std::vector<T> out(n, 0);

    for (auto j=0; j < n; ++j) {
        for (auto k=0; k < m; ++k) {
            out[j] += M[j][k] * v[k];
        }
    }
    return out;
}

template <class T>
std::vector<T> vector_matrix(std::vector<T> const &v, std::vector<std::vector<T>> const &M) {
    const int n = M.size();    // M rows
    const int m = M[0].size(); // M cols
    std::vector<T> out(m, 0);

    for (auto j=0; j < m; ++j) {
        for (auto k=0; k < n; ++k) {
            out[j] += v[k] * M[k][j];
        }
    }
    return out;
}


template <class T>
std::vector<std::vector<T>> matrix_constant(std::vector<std::vector<T>> const &A, double const &c) {
    const int n = A.size();     // A rows
    const int m = A[0].size();  // A cols
    std::vector<std::vector<T>> B(n, std::vector<T>(m, 0));

    for (auto j=0; j < n; ++j) {
        for (auto i=0; i < m; ++i) {
            B[i][j] = A[i][j] * c;
        }
    }
    return B;
}

template <class T>
std::vector<std::vector<T>> matrix_constant(double const &c, std::vector<std::vector<T>> const &A) {
    const int n = A.size();     // A rows
    const int m = A[0].size();  // A cols
    std::vector<std::vector<T>> B(n, std::vector<T>(m, 0));

    for (auto j=0; j < n; ++j) {
        for (auto i=0; i < m; ++i) {
            B[i][j] = A[i][j] * c;
        }
    }
    return B;
}

#endif // KALMAN_H