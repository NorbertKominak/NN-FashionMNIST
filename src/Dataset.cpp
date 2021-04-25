#include <fstream>
#include <sstream>
#include <algorithm>
#include "Dataset.h"

using namespace std;

Dataset::Dataset() {
}

Dataset::Dataset(string dataFile, string labelFile) {
	// this will be slow in debug mode; if the ~40s in release mode is a problem on aisa, I can make it singnificantly faster
	ifstream dataStream(dataFile);
	ifstream labelStream(labelFile);
	vector<double> data;
	data.reserve(1000);
	
	while (dataStream.good() && labelStream.good()) {
		data.clear();
		int label;
		labelStream >> label;
		labelStream.ignore();

		string dataLine;

		getline(dataStream, dataLine);
		replace(dataLine.begin(), dataLine.end(), ',', ' ');
		stringstream lineStream(dataLine);
		double d;
		while (lineStream >> d) {
			data.push_back(d/255.0);
		}

		if (data.size() > 1) {
			// workaround for empty lines; I suppose we won't need 1-feature datasets
			DataPoint dp(data, label);
			addDataPoint(dp);
		}
	}
}

void Dataset::addDataPoint(DataPoint& p) {
	dataPoints.emplace_back(p);
}

DataPoint& Dataset::operator[](int i) {
	return dataPoints[i];
}

size_t Dataset::size() {
	return dataPoints.size();
}
