#pragma once

#include "includes.cuh"
#include "activation.cuh"

namespace DeepNeuralNetwork {
	class Network;

	namespace Utils {
		__global__ void feedForward(Network* nn);
		__global__ void backPropRecord(Network* nn);
		__global__ void backPropUpdate(Network* nn);
	}

	struct Layer {
		size_t height;
		Activation::Activator activator;

		Layer(size_t height, Activation::Activator activator = Activation::Activator::None);
	};

	class Network {
		float learningRate;
		size_t layerCount;
		size_t inputHeight;
		size_t outputHeight;
		size_t tallestLayerSize;
		size_t* shape;
		Activation::Activator* activators;
		Network* this_gpuCopy;

		size_t trainBatchSize;
		float** inputs;
		float** expectedOutputs;

		float*** weights;
		float** biases;

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
		void train(float** inputs, float** outputs);
		void summary(bool showParameterValues = true);

		~Network();

		friend __global__ void Utils::feedForward(Network*);
		friend __global__ void Utils::backPropRecord(Network*);
		friend __global__ void Utils::backPropUpdate(Network*);
	};
}