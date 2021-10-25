#include "network.cuh"
namespace DNN = DeepNeuralNetwork;

DNN::Layer::Layer(size_t height, Activation::Activator activator)
{
	this->height = height;
	this->activator = activator;
}

DNN::Network::Network(
	std::initializer_list<Layer> layers,
	size_t inputHeight,
	size_t trainBatchSize,
	float learningRate,
	float*** weights,
	float** biases
)
{
	int alloc_error = 0;

	this->inputHeight = inputHeight;
	this->trainBatchSize = trainBatchSize;
	this->tallestLayerSize = 0; // set during layer loop

	this->layerCount = layers.size();

	//this->shape[layer]
	//this->activators[layer]
	alloc_error |= (int)cudaMallocManaged(&this->shape, this->layerCount * sizeof(size_t));
	alloc_error |= (int)cudaMallocManaged(&this->activators, this->layerCount * sizeof(Activation::Activator));

	//this->inputs[batch][input_idx]
	//this->unactivatedOutputs[batch][layer][node]
	//this->activatedOutputs[batch][layer][node]
	alloc_error |= (int)cudaMallocManaged(&this->inputs, this->trainBatchSize * sizeof(float*));
	alloc_error |= (int)cudaMallocManaged(&this->unactivatedOutputs, this->trainBatchSize * sizeof(float**));
	alloc_error |= (int)cudaMallocManaged(&this->activatedOutputs, this->trainBatchSize * sizeof(float**));
	for (size_t batch = 0; batch < this->trainBatchSize; batch++)
	{
		alloc_error |= (int)cudaMallocManaged(this->inputs + batch, this->inputHeight * sizeof(float));
		alloc_error |= (int)cudaMallocManaged(this->unactivatedOutputs + batch, this->layerCount * sizeof(float*));
		alloc_error |= (int)cudaMallocManaged(this->activatedOutputs + batch, this->layerCount * sizeof(float*));

		for (size_t i = 0; i < this->layerCount; i++)
		{
			const Layer* layer = layers.begin() + i;
			alloc_error |= (int)cudaMallocManaged(this->unactivatedOutputs[batch] + i, layer->height * sizeof(float*));
			alloc_error |= (int)cudaMallocManaged(this->activatedOutputs[batch] + i, layer->height * sizeof(float*));
		}
	}

	//this->weights[layer][node][prev_node]
	//this->biases[layer][node]
	alloc_error |= (int)cudaMallocManaged(&this->weights, this->layerCount * sizeof(float**));
	alloc_error |= (int)cudaMallocManaged(&this->biases, this->layerCount * sizeof(float*));
	size_t previousLayerHeight = inputHeight;
	for (size_t i = 0; i < this->layerCount; i++)
	{
		const Layer* layer = layers.begin() + i;
		this->shape[i] = layer->height;
		this->activators[i] = layer->activator;
		this->tallestLayerSize = layer->height > this->tallestLayerSize ? layer->height : this->tallestLayerSize;

		alloc_error |= (int)cudaMallocManaged(this->weights + i, layer->height * sizeof(float*));
		alloc_error |= (int)cudaMallocManaged(this->biases + i, layer->height * sizeof(float));

		for (size_t j = 0; j < layer->height; j++)
		{
			alloc_error |= (int)cudaMallocManaged(this->weights[i] + j, previousLayerHeight * sizeof(float));

			for (size_t k = 0; k < previousLayerHeight; k++)
				this->weights[i][j][k] = weights ? weights[i][j][k] : randf<-1, 1>() / layer->height;

			this->biases[i][j] = biases ? biases[i][j] : randf<-1, 1>();
		}

		previousLayerHeight = layer->height;
	}

	//this->this_gpuCopy = this (but allocated on the gpu)
	alloc_error |= (int)cudaMallocManaged(&this->this_gpuCopy, sizeof(Network));
	*this->this_gpuCopy = *this;

	this->predictionResult = this->activatedOutputs[0][this->layerCount - 1];

	if (alloc_error)
	{
		printf("There was at least one error while allocating memory. The bit mask of all errors is: %d\n", alloc_error);
		exit(1);
	}
}

void DNN::Network::predict(float* inputs)
{
	memcpy(this->inputs[0], inputs, this->inputHeight * sizeof(float));
	DNN::Utils::feedForward<<<1, this->tallestLayerSize>>>(this->this_gpuCopy);
	cudaDeviceSynchronize();
}

DNN::Network::~Network()
{
	//this->this_gpuCopy = this (but allocated on the gpu)
	cudaFree(this->this_gpuCopy);

	//this->biases[layer][node]
	//this->weights[layer][node][prev_node]
	for (size_t layer = 0; layer < this->layerCount; layer++)
	{
		for (size_t node = 0; node < this->shape[node]; node++)
		{
			cudaFree(this->weights[layer][node]);
		}
		cudaFree(this->biases[layer]);
		cudaFree(this->weights[layer]);
	}
	cudaFree(this->biases);
	cudaFree(this->weights);

	//this->activatedOutputs[batch][layer][node]
	//this->unactivatedOutputs[batch][layer][node]
	//this->inputs[batch][input_idx]
	for (size_t batch = 0; batch < this->trainBatchSize; batch++)
	{
		for (size_t layer = 0; layer < this->layerCount; layer++)
		{
			cudaFree(this->activatedOutputs[batch][layer]);
			cudaFree(this->unactivatedOutputs[batch][layer]);
		}
		cudaFree(this->activatedOutputs[batch]);
		cudaFree(this->unactivatedOutputs[batch]);
		cudaFree(this->inputs[batch]);
	}
	cudaFree(this->activatedOutputs);
	cudaFree(this->unactivatedOutputs);
	cudaFree(this->inputs);

	//this->activators[layer]
	//this->shape[layer]
	cudaFree(this->activators);
	cudaFree(this->shape);
}

__global__ void DNN::Utils::feedForward(Network* nn)
{
	const size_t& batch = blockIdx.x;
	const size_t& node = threadIdx.x;
	size_t layer = 0;

	while (layer < nn->layerCount)
	{
		// perform feedforward
		if (node < nn->shape[layer])
		{
			const Activation::Activator& activator = nn->activators[layer];
			
			float* unactivatedOutput = nn->unactivatedOutputs[batch][layer] + node;
			*unactivatedOutput = 0;

			if (layer == 0)
			{
				for (size_t j = 0; j < nn->inputHeight; j++)
				{
					*unactivatedOutput += nn->inputs[batch][j] * nn->weights[0][node][j] + nn->biases[0][node];
				}
			}
			else
			{
				const size_t& prevLayer = layer - 1;
				const size_t& prevLayerHeight = nn->shape[prevLayer];
				for (size_t j = 0; j < prevLayerHeight; j++)
				{
					*unactivatedOutput += nn->activatedOutputs[batch][prevLayer][j] * nn->weights[layer][node][j] + nn->biases[layer][node];
				}
			}

			Activation::activate(activator, *unactivatedOutput, nn->activatedOutputs[batch][layer] + node);
		}
		
		// sync everything up and move on to next layer
		__syncthreads();
		layer++;
	}
}
