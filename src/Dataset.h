#pragma once
#include <string>

#include "DataPoint.h"


class Dataset {
	std::vector<DataPoint> dataPoints;
public:
	Dataset();
	Dataset(std::string dataFile, std::string labelFile);
	void addDataPoint(DataPoint& p);
	DataPoint& operator[](int i);
	size_t size();
};