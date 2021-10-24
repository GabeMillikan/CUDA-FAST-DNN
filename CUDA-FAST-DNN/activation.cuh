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
}