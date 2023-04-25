/** 
 *  @file   test_knn.cc
 *  @brief  A test file used for check the functionalities of the KNN class.
 *  @author Stefano Leonardi
 ***********************************************/
// usage: cls && ./compile.sh && ./build/buildLinux/knn
#include "knn.hh"

/** \brief  Main function of this test.
 *  \return None
*/
int main(int argc, char * argv[]) {
    // test only code :D

    // 1° phase of training
    KNN knn = KNN();
    vector<vector<double> > pts = {{1.0, 1.0, 2}, {2.0, 2.0, 6}, {4.5, 9.0, 4}, {3.2, 1.5, 9}};
    vector<int> lbs = {1, 2, 2, 1};
    knn.train(pts, lbs);
    knn.trainWithOne({5.0, 3.0, 6}, 2);

    // print and check neighbours points
    vector<vector<double> > nPoints;
    knn.neighbourPoints({6.0, 6.0, 6.0}, nPoints);
    cout << "test1 - nPoints: " << nPoints << endl << endl;


    // 2° phase of training
    knn.trainWithOne({7.34, 8.2, 10}, 3);
    knn.trainWithOne({6, -1, 5}, 3);

    // print and check neighbours labels
    vector<int> nLabels;
    knn.neighbourLabels({6.0, 6.0, 6.0}, nLabels);
    cout << "test2 - nLabels: " << nLabels << endl << endl;
    // 3° predict the label of the new point according to the surraunding labels

    int prediction = knn.predict({6.0, 6.0, 6.0});
    cout << "test3 - The predicted label is: " << prediction << endl << endl;

    return(0);
}