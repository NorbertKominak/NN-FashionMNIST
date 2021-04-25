#include <cstdlib>

#include "Network.h"

using namespace std;

void Network::connectNeurons(int layerFrom, int neuronFrom, int layerTo, int neuronTo) {
	Neuron_ptr& from = neurons[layerFrom][neuronFrom];
	Neuron_ptr& to = neurons[layerTo][neuronTo];
	auto edge = make_shared<Edge>(from, to, (double) (layerTo / 3.0) * neurons[layerFrom].size());
	from->addOutgoingEdge(edge);
	to->addIncomingEdge(edge);
}

Network::Network(const std::vector<int>& layerSizes) {
	bias = make_shared<Neuron>();
	for (int i = 0; i < BATCH_SIZE; i++) {
		bias->value[i] = 1;
	}
	layers = layerSizes.size();
	for (int i = 0; i < BATCH_SIZE; i++) {
		expectedOutput[i] = vector<double>(layerSizes[layers - 1], 0);
	}
	for (int i = 0; i < layers; i++) {
		neurons.emplace_back();
		for (int j = 0; j < layerSizes[i]; j++) {
			neurons[i].emplace_back(make_shared<Neuron>());
			if (i > 0) {
				// add bias to the neuron
				Neuron_ptr& to = neurons[i][j];
				auto edge = make_shared<Edge>(bias, to, (i / 3.0) * neurons[i - 1].size());
				bias->addOutgoingEdge(edge);
				to->addIncomingEdge(edge);
				// add edges to the previous layer
				for (int k = 0; k < layerSizes[i - 1]; k++) {
					if (rand() % 3 < i) {
						connectNeurons(i - 1, k, i, j);
					}
				}
			}
		}
	}
}

void Network::setInputs(const std::vector<double>& input, int bid, bool noise) {
	for (size_t i = 0; i < neurons[0].size(); ++i) {
		neurons[0][i]->value[bid] = input[i];
		if (noise) {
			neurons[0][i]->value[bid] += ((rand() % 200) - 100ll) / 1000.0;
		}
	}
}

void Network::evaluate(int bid) {
	for (int i = 1; i < layers; i++) {
		for (auto& neuron : neurons[i]) {
			neuron->updateValue(bid);
		}
	}

	// Output layer evaluation using softmax, in each output neuron->value is stored potential[bid]
	double max = -DBL_MAX;
	for (auto& neuron : neurons[neurons.size() - 1]) {
		if (neuron->value[bid] > max) {
			max = neuron->value[bid];
		}
	}

	double expSum = 0;
	for (auto& neuron : neurons[neurons.size() - 1]) {
		neuron->value[bid] = exp(neuron->value[bid] - max);
		expSum += neuron->value[bid];
	}

	for (auto& neuron : neurons[neurons.size() - 1]) {
		neuron->value[bid] = neuron->activationOutput(neuron->value[bid], expSum);
	}
}

void Network::backprop(double learningRate, double momentumRate) {
	// assuming that the gradients from the output layer are already known
	for (int i = layers - 1; i >= 0; i--) {
		for (auto& neuron : neurons[i]) {
			if (i < layers - 1) {
				neuron->updateGradient();
				neuron->updateWeights(learningRate, momentumRate);
			}
			if (i > 0) {
				neuron->updateBiasWeight(learningRate, momentumRate);
			}
		}
	}
}

void Network::setOutputLayerGradient() {
	int lastLayerIndex = neurons.size() - 1;
	for (size_t i = 0; i < neurons[lastLayerIndex].size(); ++i) {
		for (int j = 0; j < BATCH_SIZE; j++) {
			auto& n = *(neurons[lastLayerIndex][i]);
			n.gradient[j] = (n.value[j] - expectedOutput[j][i]);
			n.actigrad[j] = n.gradient[j] * n.activationOutputDerivative(n.value[j]);
		}
	}
}

void Network::setExpected(std::vector<double>& expected, int bid) {
	expectedOutput[bid] = expected;
}

vector<double> Network::getOutputs(int bid) {
	vector<double> result;
	result.reserve(neurons[neurons.size() - 1].size());
	for (size_t i = 0; i < neurons[neurons.size() - 1].size(); ++i) {
		result.push_back(neurons[neurons.size() - 1][i]->value[bid]);
	}
	return result;
}