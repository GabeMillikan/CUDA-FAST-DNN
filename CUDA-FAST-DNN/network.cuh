#pragma once

#include "includes.cuh"
#include "activation.cuh"

namespace DeepNeuralNetwork {
	class Network;

	namespace Utils {
		__global__ void feedForward(Network* nn);
	}

	struct Layer {
		size_t height;
		Activation::Activator activator;

		Layer(size_t height, Activation::Activator activator = Activation::Activator::None);
	};

	class Network {
		size_t layerCount;
		size_t inputHeight;
		size_t tallestLayerSize;
		size_t* shape;
		Activation::Activator* activators;
		Network* this_gpuCopy;

		float** inputs;
		float*** weights;
		float** biases;

		size_t trainBatchSize;
		float*** unactivatedOutputs;
		float*** activatedOutputs;
	
	public:
		float* predictionResult; // READ ONLY!!

		Network(
			std::initializer_list<Layer> layers,
			size_t inputHeight,
			size_t trainBatchSize,
			float learningRate = 0.001f,
			float*** weights = nullptr,
			float** biases = nullptr
		);

		void predict(float* inputs);

		~Network();

		friend __global__ void Utils::feedForward(Network*);
	};
}