#pragma once
#include "includes.cuh"

namespace Activation
{
	enum class Activator
	{
		None = 0,
		Linear = 0,
		ReLu,
		Sigmoid
	};


	const char* stringifyActivator(const Activator& activator);

	__device__ void activate(const Activator& activator, const float& in, float* out);
	__device__ void differentiate(const Activator& activator, const float& in, float* out);
}