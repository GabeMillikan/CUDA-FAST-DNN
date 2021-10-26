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

	this->learningRate = learningRate;
	this->inputHeight = inputHeight;
	this->trainBatchSize = trainBatchSize;
	this->tallestLayerSize = 0; // set during layer loop

	this->layerCount = layers.size();
	
	this->outputHeight = (layers.begin() + this->layerCount - 1)->height;

	//this->shape[layer]
	//this->activators[layer]
	alloc_error |= (int)cudaMallocManaged(&this->shape, this->layerCount * sizeof(size_t));
	alloc_error |= (int)cudaMallocManaged(&this->activators, this->layerCount * sizeof(Activation::Activator));

	//this->inputs[batch][input_idx]
	//this->expectedOutputs[batch][output_idx]
	//this->unactivatedOutputs[batch][layer][node]
	//this->activatedOutputs[batch][layer][node]
	alloc_error |= (int)cudaMallocManaged(&this->inputs, this->trainBatchSize * sizeof(float*));
	alloc_error |= (int)cudaMallocManaged(&this->expectedOutputs, this->trainBatchSize * sizeof(float*));
	alloc_error |= (int)cudaMallocManaged(&this->unactivatedOutputs, this->trainBatchSize * sizeof(float**));
	alloc_error |= (int)cudaMallocManaged(&this->activatedOutputs, this->trainBatchSize * sizeof(float**));
	for (size_t batch = 0; batch < this->trainBatchSize; batch++)
	{
		alloc_error |= (int)cudaMallocManaged(this->inputs + batch, this->inputHeight * sizeof(float));
		alloc_error |= (int)cudaMallocManaged(this->expectedOutputs + batch, this->outputHeight * sizeof(float));
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

void DNN::Network::train(float** inputs, float** outputs)
{
	for (size_t batch = 0; batch < this->trainBatchSize; batch++)
	{
		memcpy(this->inputs[batch], inputs[batch], this->inputHeight * sizeof(float));
		memcpy(this->expectedOutputs[batch], outputs[batch], this->outputHeight * sizeof(float));
	}

	DNN::Utils::feedForward<<<this->trainBatchSize, this->tallestLayerSize>>>(this->this_gpuCopy);
	cudaDeviceSynchronize();

	DNN::Utils::backPropRecord<<<this->trainBatchSize, this->tallestLayerSize>>>(this->this_gpuCopy);
	cudaDeviceSynchronize();
	
	DNN::Utils::backPropUpdate<<<this->layerCount, this->tallestLayerSize>>>(this->this_gpuCopy);
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
	//this->expectedOutputs[batch][output_idx]
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
		cudaFree(this->expectedOutputs[batch]);
		cudaFree(this->inputs[batch]);
	}
	cudaFree(this->activatedOutputs);
	cudaFree(this->unactivatedOutputs);
	cudaFree(this->expectedOutputs);
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

__global__ void DNN::Utils::backPropRecord(Network* nn)
{
	/*
	These equations are followed religiously:
	https://cdn.discordapp.com/attachments/282239317966061568/902381475549282354/342191494872039424.png

	o_{xi} = \sum_{j=1}^{I_{x-1}}w_{xij}a_{(x-1)j}+b_{xi} \\
	a_{xi} = \sigma(o_{xi}) \\
	C = \frac{1}{I}\sum_{i=1}^{I_X}(a_{Xi}-t_i)^2 \\
	\frac{dC}{do_{Xi}} = \frac{2}{I_X}(a_{Xi} - t_{i})\sigma ^ \prime (o_{Xi}) \\
	\frac{dC}{do_{xi}} = \sum_{j=1}^{I_{x+1}}\frac{dC}{do_{(x+1)j}} \cdot w_{(x+1)ji} \cdot \sigma ^\prime (o_{xi}) \\
	\frac{do_{xi}}{dw_{xij}} = a_{(x-1)j} \\
	\frac{do_{xi}}{db_{xi}} = 1 \\
	\frac{dC}{dw_{xij}} = \frac{dC}{do_{xi}}\cdot \frac{do_{xi}}{dw_{xij}} = \frac{dC}{do_{xi}}\cdot a_{(x-1)j}\\
	\frac{dC}{db_{xi}} = \frac{dC}{do_{xi}}\cdot \frac{do_{xi}}{db_{xi}} = \frac{dC}{do_{xi}}

	nn->unactivatedOutputs will be used to store dc/do
	*/

	const size_t& batch = blockIdx.x;
	const size_t& node = threadIdx.x;

	// calculate dc/do for the very last layer
	size_t layer = nn->layerCount - 1;
	if (node < nn->outputHeight)
	{
		float* o_Xi = nn->unactivatedOutputs[batch][layer] + node;
		const float err = *o_Xi - nn->expectedOutputs[batch][node];

		Activation::differentiate(nn->activators[layer], *o_Xi, o_Xi);
		*o_Xi *= 2.f * err / nn->outputHeight;
	}

	// now for every other layer
	--layer;
	size_t layerHeight = nn->shape[layer], followingLayerHeight = nn->outputHeight, j = 0;
	while (layer != std::numeric_limits<size_t>::max())
	{
		__syncthreads();
		if (node < layerHeight)
		{
			float* o_xi = nn->unactivatedOutputs[batch][layer] + node;
			Activation::differentiate(nn->activators[layer], *o_xi, o_xi);

			++layer;
			float j_sum = 0.f;
			for (j = 0; j < followingLayerHeight; ++j)
				j_sum += nn->unactivatedOutputs[batch][layer][j] * nn->weights[layer][j][node];
			--layer;

			*o_xi *= j_sum;
			//*a_xi *= j_sum / followingLayerHeight; // divide just to normalize into reasonable range
		}

		--layer;
		followingLayerHeight = layerHeight;
		layerHeight = nn->shape[layer];
	}

	// now, all of the dc/do information is stored in nn->activatedOutputs
	// next: call backPropUpdate()
}


void DNN::Utils::backPropUpdate(Network* nn)
{
	const size_t& layer = blockIdx.x;
	const size_t& node = threadIdx.x;
	const size_t& batchSize = nn->trainBatchSize;
	const size_t& prevLayerHeight = layer == 0 ? nn->inputHeight : nn->shape[layer - 1];

	if (nn->shape[layer] <= node) return;

	float avg_dc_do = 0.f;
	for (size_t batch = 0; batch < batchSize; ++batch)
	{
		avg_dc_do += nn->activatedOutputs[batch][layer][node];
	}
	avg_dc_do /= batchSize;

	// weights
	for (size_t weight = 0; weight < prevLayerHeight; weight++)
	{
		nn->weights[layer][node][weight] -= avg_dc_do * weight * nn->learningRate;
	}
}