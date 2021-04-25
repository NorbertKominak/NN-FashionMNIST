#include <cstdint>
#include <random>

#include "Edge.h"
#include "Neuron.h"

Edge::Edge(Neuron_ptr& _from, Neuron_ptr& _to, double inputsSize)
	: from(_from), to(_to), weight(initializeWeight(inputsSize)), prevWChange(0) { 
}

double Edge::initializeWeight(double inputsSize) {
	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::normal_distribution<> d{ 0.0, sqrt(2.0 / inputsSize) };
	return d(gen);
}

double Edge::value(int bid) {
	return from->value[bid] * weight;
}
