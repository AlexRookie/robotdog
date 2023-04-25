/** 
 *  @file   knn.hh
 *  @brief  A file containing the implementation of the my version of the KNN algorithm.
 *  @author Stefano Leonardi
 ***********************************************/
#pragma once
#include <vector>
#include <utility>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <map>

#include "utils.hh"
using std::cout; 
using std::endl;


/*! An enum used to specify the possible labels of the KNN class. */
enum KnnLabels{SOMEONE=0, LEADER=1, UNKNOWN=2};

// The class can be easily exted to manage any kind of labels (not only int) with templates
// To do so the .hh and .cc files should be collapsed into a single one...
// No need at the moment so it has not been implemented.

/*! The KNN class is a standard implementation of the classical, and very well known, algorithm, the uniqueness of this code is that knn can be trained multiple times and all the samples are treated equally.*/
class KNN{
    private:
        vector<vector<float> > knnPoints;
        vector<int> knnLabels;

        /** \brief Compute the indexes of the k closest points to the given query (AKA the neighbours).
         *  \param[in] query The query (AKA a point) that will be used to search the neighbours.
         *  \param[out] labels The vector containing the indexes of the k neighbour points.
         *  \param[in] k How many neighbours should be located.
        */
        void neighboursIndexes(const vector<float> & query, vector<int> & indexes, vector<double> & dists, int k=5);

        /** \brief Look for the most frequent element inside the vector. (linear scan)
         *  \param[in] v The vector to be scan.
         *  \return The most frequent element.
        */
        int mostFrequentElement(const vector<int> & v);

    public:
        /** \brief The constructor method of the class.*/
        KNN();

        /** \brief KNN is trained with the given values, AKA these values are stored.
         *  \param[in] points The list of points that will be stored inside knn. Each point is represented as a vector of float.
         *  \param[in] labels The list of labels that will be stored inside knn. Each label is an int value.
        */
        void train(const vector<vector<float> > & points, const vector<int> & labels);

        /** \brief KNN is trained with the given value, AKA the value is stored.
         *  \param[in] point The point that will be stored inside knn.
         *  \param[in] label The label that will be stored inside knn.
        */
        void trainWithOne(const vector<float> & point, int label);
        
        /** \brief A print function that will show the labels and the points available.*/
        void print();


        /** \brief Compute the k closest points to the given query (AKA the neighbours).
         *  \param[in] query The query (AKA a point) that will be used to search the neighbours.
         *  \param[out] points The vector containing the k neighbour points.
         *  \param[in] k How many neighbours should be located.
        */
        void neighbourPoints(const vector<float> & query, vector<vector<float> > & points, int k=5);
        
        /** \brief Compute the labels of the k closest points to the given query (AKA the neighbours).
         *  \param[in] query The query (AKA a point) that will be used to search the neighbours.
         *  \param[out] labels The vector containing the labels of the k neighbour points.
         *  \param[in] k How many neighbours should be located.
        */
        void neighbourLabels(const vector<float> & query, vector<int> & labels, int k=5);


        /** \brief Predict a new label for the unknown point.
         *  \param[in] newPoint The unknown point that should be classified.
         *  \param[in] k How many neighbours are use to perform the prediction.
         *  \return The predicted label.
        */
        int predict(const vector<float> & newPoint, double & dst, int & cnt, int k=5);
}; 
