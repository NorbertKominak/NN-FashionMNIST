#pragma once
#include <memory>
#include <vector>

#define BATCH_SIZE 30

#include "Edge.h"

using Edge_ptr = std::shared_ptr<Edge>;

class Neuron {

	std::vector<Edge_ptr> incoming;
	std::vector<Edge_ptr> outgoing;

public:
	double activation(double x);
	double activationOutput(double x, double sum);
	double activationDerivative(double x);
	double activationOutputDerivative(double x);
	void addIncomingEdge(Edge_ptr& edge);
	void addOutgoingEdge(Edge_ptr& edge);
	void updateGradient();
	void updateWeights(double learningRate, double momentumRate);
	void updateBiasWeight(double learningRate, double momentumRate);
	double gradient[BATCH_SIZE];
	double potential[BATCH_SIZE];
	double value[BATCH_SIZE];
	double actigrad[BATCH_SIZE];
	void updateValue(int bid);
	Neuron();

};