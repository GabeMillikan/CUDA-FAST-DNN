#include "includes.cuh"
#include "network.cuh"
namespace DNN = DeepNeuralNetwork;

int main()
{
	auto network = DNN::Network({ 1 }, 1);

	float inputs[] = {
		5.f
	};
	printf("bias before feedforward: %.3f\n", network.biases[0][0]);
	network.feedForward(inputs);
	printf("bias after feedforward: %.3f\n", network.biases[0][0]);
}