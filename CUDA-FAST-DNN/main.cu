#include "includes.cuh"
#include "network.cuh"
namespace DNN = DeepNeuralNetwork;

int main()
{
	auto network = DNN::Network({ 1 }, 1, 1, 0.1);
	network.summary();

	for (int i = 0; i <= 100; i++)
	{
		float in = randf<-1, 1>();
		float* pIn = &in;

		printf("\nTRAIN STEP %d - INPUT = %.4f\n", i, in);

		network.train(&pIn, &pIn);
		network.summary();
	}


}