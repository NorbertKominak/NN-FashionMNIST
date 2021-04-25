#include <algorithm>
#include <numeric>
#include <iostream>

#include "Neuron.h"

using namespace std;

double Neuron::activation(double x) {
	// leaky ReLU
	return x * (x > 0 ? 1 : 0.01);
}

double Neuron::activationOutput(double x, double sum) {
	// Softmax
	return x / sum;
}

double Neuron::activationDerivative(double x)
{
	return (x > 0 ? 1 : 0.01);
}

double Neuron::activationOutputDerivative(double x) {
	return x * (1 - x);
}

void Neuron::addIncomingEdge(Edge_ptr& edge) {
	incoming.emplace_back(edge);
}

void Neuron::addOutgoingEdge(Edge_ptr& edge) {
	outgoing.emplace_back(edge);
}

void Neuron::updateGradient() {
	for (int i = 0; i < BATCH_SIZE; i++) {
		gradient[i] = 0;
		for (auto& e : outgoing) {
			auto& edge = *e;
			auto& n = *(edge.to);
			gradient[i] += n.actigrad[i] * edge.weight;
		}
		actigrad[i] = gradient[i] * activationDerivative(potential[i]);
	}
}

const double smoothTerm = 1e-8;

void Neuron::updateWeights(double learningRate, double momentumRate) {
	
	for (auto& e : outgoing) {
		double adsum = 0;
		auto& n = *(e->to);
		for (int i = 0; i < BATCH_SIZE; i++) {
			adsum += value[i] * n.actigrad[i];
		}
		double currentWChange = -learningRate * adsum + momentumRate * e->prevWChange;
		e->weight += currentWChange;	
		e->prevWChange = currentWChange;
	}
}

void Neuron::updateBiasWeight(double learningRate, double momentumRate) {
	auto& edge = incoming[0];
	double adsum = 0;
	for (int i = 0; i < BATCH_SIZE; i++) {
		adsum += actigrad[i];
	}
	double currentWChange = -learningRate * adsum + momentumRate * edge->prevWChange;
	edge->weight += currentWChange;
	edge->prevWChange = currentWChange;
}

void Neuron::updateValue(int bid) {
	potential[bid] = 0;
	for (auto& e : incoming) {
		potential[bid] += e->value(bid);
	}

	value[bid] = outgoing.size() == 0 ? potential[bid] : activation(potential[bid]);
}

Neuron::Neuron() {
}
