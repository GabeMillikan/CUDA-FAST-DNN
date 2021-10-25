#include "activation.cuh"

__device__ void Activation::activate(const Activator& activator, const float& in, float* out)
{
	switch (activator)
	{
	default:
	case Activator::Linear:
		*out = in;
		break;
	case Activator::ReLu:
		*out = in <= 0 ? 0 : in;
		break;
	case Activator::Sigmoid:
		*out = 1 / (1 + exp(-in));
		break;
	}
}

/*
double Activation::differentiate(const Activator& activator, const double& input)
{
	switch (activator)
	{
	default:
	case Activator::Linear:
		return 1;
	case Activator::ReLu:
		return input <= 0 ? 0 : 1;
	case Activator::Sigmoid:
		double sigmoid = activate(activator, input);
		return sigmoid * (1 - sigmoid);
	}
}
*/