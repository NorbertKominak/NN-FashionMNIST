#pragma once
#include <cmath>
#include <cfloat>

#include "Neuron.h"


class Network {
	int layers = 0;
	std::vector<std::vector<Neuron_ptr>> neurons;
	void connectNeurons(int layerFrom, int neuronFrom, int layerTo, int neuronTo);
	Neuron_ptr bias;
	std::vector<double> expectedOutput[BATCH_SIZE];
public:
	// constructs a complete NN, with given number of neurons in every layer
	Network(const std::vector<int>& layers);
	void setInputs(const std::vector<double>& input, int bid, bool noise);
	void evaluate(int bid);
	void backprop(double learningRate, double momentumRate);
	void setOutputLayerGradient();
	void setExpected(std::vector<double>& expected, int bid);
	std::vector<double> getOutputs(int bid);
};