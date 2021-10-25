#include "includes.cuh"
#include "network.cuh"
namespace DNN = DeepNeuralNetwork;

int main()
{
	auto network = DNN::Network({ 2 }, 2, 16);

	float inputs[] = {
		5.f,
		-5.f
	};
	network.predict(inputs);

	printf("(%.3f, %.3f) -> (%.3f, %.3f)\n", inputs[0], inputs[1], network.predictionResult[0], network.predictionResult[1]);
}