#include "activation.cuh"

const char* Activation::stringifyActivator(const Activator& activator)
{
	switch (activator)
	{
	default:
	case Activator::Linear:
		return "Linear";
	case Activator::ReLu:
		return "ReLu";
	case Activator::Sigmoid:
		return "Sigmoid";
	}
}

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

__device__ void Activation::differentiate(const Activator& activator, const float& in, float* out)
{
	switch (activator)
	{
	default:
	case Activator::Linear:
		*out = 1.f;
		break;
	case Activator::ReLu:
		*out = in <= 0 ? 0 : 1;
		break;
	case Activator::Sigmoid:
		activate(activator, in, out);
		*out *= (1 - *out);
		break;
	}
}
