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
	this->shape = new size_t[this->layerCount];
	this->activators = new Activation::Activator[this->layerCount];

	this->unactivatedOutputs = new float* [this->layerCount];
	this->activatedOutputs = new float* [this->layerCount];

	this->weights = new float** [this->layerCount];
	this->biases = new float* [this->layerCount];

	size_t previousLayerHeight = inputHeight;
	for (size_t i = 0; i < this->layerCount; i++)
	{
		const Layer* layer = layers.begin() + i;
		this->shape[i] = layer->height;
		this->activators[i] = layer->activator;

		this->unactivatedOutputs[i] = new float[layer->height];
		this->activatedOutputs[i] = new float[layer->height];

		this->weights[i] = new float* [layer->height];
		this->biases[i] = new float[layer->height];

		for (size_t j = 0; j < layer->height; j++)
		{
			this->weights[i][j] = new float[previousLayerHeight];
			for (size_t k = 0; k < previousLayerHeight; k++)
				this->weights[i][j][k] = weights ? weights[i][j][k] : randf<-1, 1>();

			this->biases[i][j] = biases ? biases[i][j] : randf<-1, 1>();
		}

		previousLayerHeight = layer->height;
	}
}

void DNN::Network::feedForward(float* inputs)
{

}

DNN::Network::~Network()
{
	for (size_t i = 0; i < this->layerCount; i++)
	{
		for (size_t j = 0; j < this->shape[i]; j++)
		{
			delete[] this->weights[i][j];
		}

		delete[] this->biases[i];
		delete[] this->weights[i];
	}
	delete[] this->biases;
	delete[] this->weights;

	delete[] this->shape;
	delete[] this->activators;
}