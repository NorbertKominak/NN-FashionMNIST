#pragma once
#include <vector>

class DataPoint {
public:
	std::vector<double> features;
	int label;
	bool success = false;

	DataPoint(const std::vector<double>& _features, int _label) 
		: features(_features), label(_label) {}
	// "features(_features)" means that features = std::vector<double>(_features)

	// operator overloading: you can use "DataPoint[5]" instead of "DataPoint.features[5]"
	double& operator[](int i) {
		return features[i];
	} 

	size_t size() {
		return features.size();
	}
};