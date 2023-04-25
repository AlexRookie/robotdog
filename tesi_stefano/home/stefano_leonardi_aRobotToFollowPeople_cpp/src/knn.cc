#include "knn.hh"

KNN::KNN(){
    this->knnPoints = {};
    this->knnLabels = {};
}

void KNN::train(const vector<vector<float> > & points, const vector<int> & labels){
    // consistency check
    if(points.size() != labels.size()){
        cout << "Error: the number of points and labels given is not equal.\n\tNo point will be added to KNN.\n";
    } else{
        // store the points internally
        for(int i=0; i<points.size(); i++){
            if (labels[i]!=UNKNOWN) {
                this->knnPoints.push_back(points[i]);
                this->knnLabels.push_back(labels[i]);
            }
        }
    }
}

void KNN::trainWithOne(const vector<float> & point, int label){
    // store the point internally
    if (label!=UNKNOWN) {
        this->knnPoints.push_back(point);
        this->knnLabels.push_back(label);
    }
}

void KNN::print(){
    for(int i=0; i<knnPoints.size(); i++){
        cout << i << "° " << knnLabels[i] << ": " << knnPoints[i] << endl;
    }
    cout << endl;
}

void KNN::neighbourPoints(const vector<float> & query, vector<vector<float> > & points, int k){
    vector<int> indexes;
    vector<double> dists;
    // retrieve the neighbours indexes and then retrieve associate points
    neighboursIndexes(query, indexes, dists, k);
    for(int i=0; i<indexes.size(); i++){
        points.push_back( knnPoints[indexes[i]] );
    }
}

void KNN::neighbourLabels(const vector<float> & query, vector<int> & labels, int k){
    vector<int> indexes;
    vector<double> dists;
    // retrieve the neighbours indexes and then retrieve associate labels
    neighboursIndexes(query, indexes, dists, k);
    for(int i=0; i<indexes.size(); i++){
        labels.push_back( knnLabels[indexes[i]] );
    }
}

// private
void KNN::neighboursIndexes(const vector<float> & query, vector<int> & indexes, vector<double> & dists, int k){
    int len = (int)knnPoints.size();

    // corner case. K is greater than all the available points.
    if(k>len){
        k=len;
    }

    // store both indexes and distances
    vector<std::pair<float, int> > distances;
    for(int i=0; i<len; i++){
        // I compute the distance without the root operation because is useless computation.
        float dist = (float)cv::norm(query, knnPoints[i], cv::NORM_L2SQR);
        distances.push_back( {dist, i} );
    }
    // sort the values according to the distance
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

    for(int i=0; i<k; i++){
        indexes.push_back(distances[i].second);
        dists.push_back(distances[i].first);
    }
}

int KNN::predict(const vector<float> & newPoint, double & dst, int & cnt, int k){
    std::cout << "Dataset size: " << knnPoints.size() << "\n";

    // the prediction is defined as the most frequet label of the neighbour points
    vector<int> labels;
    //neighbourLabels(newPoint, labels, k);
    vector<int> indexes;
    vector<double> dists;
    neighboursIndexes(newPoint, indexes, dists, k);
    
    for(int i=0; i<indexes.size(); i++){
        labels.push_back( knnLabels[indexes[i]] );
    }
    int label = mostFrequentElement(labels);

    dst = 0.0;
    //int cnt = 0;
    cnt = 0;
    for(int i=0; i<indexes.size(); i++){
        if (knnLabels[indexes[i]] == label) {
            dst += dists[i];
            ++cnt;
        }
    }
    if (cnt) {
        dst /= cnt;
    }

    return label;
}

// private
int KNN::mostFrequentElement(const vector<int> & v){
    int max = 0;
    int most_common = -1;
    std::map<int,int> m;
    // scan the vector linearly
    for(int i=0; i<v.size(); i++) {
        // update the map cout of the elements
        m[v[i]]++;
        if (m[v[i]] > max) {
            max = m[v[i]]; 
            most_common = v[i];
        }
    }
    return most_common;
}