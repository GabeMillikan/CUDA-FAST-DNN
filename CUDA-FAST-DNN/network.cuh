#pragma once

#include "includes.cuh"
#include "activation.cuh"

namespace DeepNeuralNetwork {
	namespace Utils {

	}

	struct Layer {
		size_t height;
		Activation::Activator activator;

		Layer(size_t height, Activation::Activator activator = Activation::Activator::None);
	};

	struct Network {
		size_t layerCount;
		size_t inputHeight;
		size_t* shape;
		Activation::Activator* activators;

		float*** weights;
		float** biases;

		float** unactivatedOutputs;
		float** activatedOutputs;
		float* networkOutputs;

		Network(
			std::initializer_list<Layer> layers,
			size_t inputHeight,
			float learningRate = 0.001f,
			float*** weights = nullptr,
			float** biases = nullptr
		);

		void feedForward(float* inputs);

		~Network();
	};
}