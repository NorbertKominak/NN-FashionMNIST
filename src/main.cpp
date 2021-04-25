#include <iostream>
#include <cstdint>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <fstream>
#include "Network.h"
#include "Dataset.h"

// VS doesn't like C-like srand setup, so this just turns off the warning in VS
#ifdef _MSC_VER
#pragma warning( disable : 4244 )
#endif

using namespace std;
const double learningRate = 0.0075;
const double momentumRate = 0.85;
const double ratesDecay = 7.0;
const int trainTimeLimit = 1500; // In seconds

int main() {
	srand(0);
	ofstream outfile;

	Dataset fashion("data/fashion_mnist_train_vectors.csv", "data/fashion_mnist_train_labels.csv");
	Dataset test("data/fashion_mnist_test_vectors.csv", "data/fashion_mnist_train_labels.csv");

	vector<int> layers_structure{ 784, 256, 64, 10 };
	vector<double> expected(10, 0);
	auto n = Network(layers_structure);
	int epoch = 0;
	int trainTime = 0;

	do {
		auto startEpochTime = chrono::high_resolution_clock::now();
		double correct = 0;

		for (size_t i = 0; i < fashion.size() / BATCH_SIZE; ++i) {
			for (int j = 0; j < BATCH_SIZE; j++) {
				int id = i * BATCH_SIZE + j;
				n.setInputs(fashion[id].features, j, fashion[id].success);
				n.evaluate(j);
				auto outputs = n.getOutputs(j);
				expected[fashion[id].label] = 1;
				if (max_element(outputs.begin(), outputs.end()) - outputs.begin() == fashion[id].label) {
					fashion[id].success = true;
					correct++;
				}
				else {
					fashion[id].success = false;
				}
				n.setExpected(expected, j);
				expected[fashion[id].label] = 0;
			}

			n.setOutputLayerGradient();
			n.backprop(learningRate / (1ll + epoch / ratesDecay), momentumRate / (1ll + epoch / ratesDecay));

		}
		cout << "  Epoch " << epoch << ": " << (100.0 * correct) / fashion.size() << "% correct" << endl;
		auto endEpochTime = chrono::high_resolution_clock::now();
		auto durationEpoch = chrono::duration_cast<chrono::seconds>(endEpochTime - startEpochTime).count();
		cout << "  Time taken by epoch: " << durationEpoch << " seconds" << endl;
		trainTime += durationEpoch;
		cout << "  Time taken by training so far: " << trainTime << " seconds" << endl;
		epoch++;
	} while (trainTime < trainTimeLimit);


	outfile.open("actualPredictions", ios::app);
	for (size_t j = 0; j < test.size(); ++j) {
		n.setInputs(test[j].features, 0, false);
		n.evaluate(0);
		auto outputs = n.getOutputs(0);
		int out = max_element(outputs.begin(), outputs.end()) - outputs.begin();
		outfile << out << endl;
	}
	
}
