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

	__device__ void activate(const Activator& activator, const float& in, float* out);
	//__host__ __device__ inline void differentiate(const Activator& activator, const float& in, float* out);
}