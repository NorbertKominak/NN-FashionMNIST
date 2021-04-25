#pragma once
#include <memory>

class Neuron;

using Neuron_ptr = std::shared_ptr<Neuron>;

class Edge {
	double initializeWeight(double inputsSize);

public:
	Neuron_ptr from;
	Neuron_ptr to;
	double weight;
	double prevWChange;
	Edge(Neuron_ptr& _from, Neuron_ptr& _to, double inputsSize);
	double value(int bid);
};