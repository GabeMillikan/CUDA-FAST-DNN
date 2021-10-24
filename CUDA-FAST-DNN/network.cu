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
	float learningRate,
	float*** weights,
	float** biases
)
{
	this->inputHeight = inputHeight;

	this->layerCount = layers.size();

	//this->inputs = new float[this->inputHeight];
	cudaMallocManaged(&this->inputs, this->inputHeight * sizeof(float));
	//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->inputs);

	//this->shape = new size_t[this->layerCount];
	//this->activators = new Activation::Activator[this->layerCount];
	cudaMallocManaged(&this->shape, this->layerCount * sizeof(size_t));
	//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->shape);
	cudaMallocManaged(&this->activators, this->layerCount * sizeof(Activation::Activator));
	//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->activators);

	//this->unactivatedOutputs = new float* [this->layerCount];
	//this->activatedOutputs = new float* [this->layerCount];
	cudaMallocManaged(&this->unactivatedOutputs, this->layerCount * sizeof(float*));
	//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->unactivatedOutputs);
	cudaMallocManaged(&this->activatedOutputs, this->layerCount * sizeof(float*));
	//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->activatedOutputs);

	//this->weights = new float** [this->layerCount];
	//this->biases = new float* [this->layerCount];
	cudaMallocManaged(&this->weights, this->layerCount * sizeof(float**));
	//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->weights);
	cudaMallocManaged(&this->biases, this->layerCount * sizeof(float*));
	//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->biases);

	size_t previousLayerHeight = inputHeight;
	for (size_t i = 0; i < this->layerCount; i++)
	{
		const Layer* layer = layers.begin() + i;
		this->shape[i] = layer->height;
		this->activators[i] = layer->activator;

		//this->unactivatedOutputs[i] = new float[layer->height];
		//this->activatedOutputs[i] = new float[layer->height];
		cudaMallocManaged(this->unactivatedOutputs + i, layer->height * sizeof(float));
		//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->unactivatedOutputs[i]);
		cudaMallocManaged(this->activatedOutputs + i, layer->height * sizeof(float));
		//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->activatedOutputs[i]);

		//this->weights[i] = new float* [layer->height];
		//this->biases[i] = new float[layer->height];
		cudaMallocManaged(this->weights + i, layer->height * sizeof(float*));
		//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->weights[i]);
		cudaMallocManaged(this->biases + i, layer->height * sizeof(float));
		//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->biases[i]);

		for (size_t j = 0; j < layer->height; j++)
		{
			//this->weights[i][j] = new float[previousLayerHeight];
			cudaMallocManaged(this->weights[i] + j, previousLayerHeight * sizeof(float));
			//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->weights[i][j]);

			for (size_t k = 0; k < previousLayerHeight; k++)
				this->weights[i][j][k] = weights ? weights[i][j][k] : randf<-1, 1>();

			this->biases[i][j] = biases ? biases[i][j] : randf<-1, 1>();
		}

		previousLayerHeight = layer->height;
	}

	this->networkOutputs = this->activatedOutputs[this->layerCount - 1];

	//this->this_gpuCopy = (Network*)malloc(sizeof(Network));
	cudaMallocManaged(&this->this_gpuCopy, sizeof(Network));
	//[AUTOCOMMENT-allocations]printf("allocated %p\n", this->this_gpuCopy);

	*this->this_gpuCopy = *this;
}

void DNN::Network::feedForward(float* inputs)
{
	memcpy(this->inputs, inputs, this->inputHeight * sizeof(float));
	DNN::Utils::feedForward<<<1, 1>>>(this->this_gpuCopy);
	cudaDeviceSynchronize();
}

DNN::Network::~Network()
{
	for (size_t i = 0; i < this->layerCount; i++)
	{
		for (size_t j = 0; j < this->shape[i]; j++)
		{
			//delete[] this->weights[i][j];
			//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->weights[i][j]);
			cudaFree(this->weights[i][j]);
		}

		//delete[] this->weights[i];
		//delete[] this->biases[i];
		//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->weights[i]);
		cudaFree(this->weights[i]);
		//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->biases[i]);
		cudaFree(this->biases[i]);

		//delete[] this->unactivatedOutputs[i];
		//delete[] this->activatedOutputs[i];
		//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->unactivatedOutputs[i]);
		cudaFree(this->unactivatedOutputs[i]);
		//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->activatedOutputs[i]);
		cudaFree(this->activatedOutputs[i]);
	}

	//delete[] this->biases;
	//delete[] this->weights;
	//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->weights);
	cudaFree(this->weights);
	//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->biases);
	cudaFree(this->biases);

	//delete[] this->unactivatedOutputs;
	//delete[] this->activatedOutputs
	//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->unactivatedOutputs);
	cudaFree(this->unactivatedOutputs);
	//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->activatedOutputs);
	cudaFree(this->activatedOutputs);

	//delete[] this->inputs;
	//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->inputs);
	cudaFree(this->inputs);

	//delete[] this->shape;
	//delete[] this->activators;
	//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->shape);
	cudaFree(this->shape);
	//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->activators);
	cudaFree(this->activators);

	//free(gpu_shallowCopy);
	//[AUTOCOMMENT-allocations]printf("freeing %p\n", this->gpu_shallowCopy);
	cudaFree(this->this_gpuCopy);
}


__global__ void DNN::Utils::feedForward(Network* nn)
{
	printf("input while feeding: %.3f\n", nn->inputs[0]);
	nn->biases[0][0] = 69.420f;
}