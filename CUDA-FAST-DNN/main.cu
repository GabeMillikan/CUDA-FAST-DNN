#include "includes.cuh"
#include "network.cuh"
namespace DNN = DeepNeuralNetwork;

int main()
{
	int batchSize = 128;
	int ioSize = 1;
	auto network = DNN::Network({ 128, ioSize }, ioSize, batchSize, 0.01f);
	network.summary();

	int trainSteps = 1000;
	int logInterval = 100;

	float** batchInfo = new float* [batchSize];
	for (size_t b = 0; b < batchSize; b++)
	{
		batchInfo[b] = new float[ioSize];
	}

	float rollingLoss = -1.f;
	float loss = 0.f;
	for (int i = 1; i <= trainSteps; i++)
	{
		for (size_t b = 0; b < batchSize; b++)
		{
			for (size_t i = 0; i < ioSize; i++)
			{
				batchInfo[b][i] = randf<-1, 1>();
			}
		}
		network.train(batchInfo, batchInfo, &loss);

		// weighted average
		if (rollingLoss < 0.f) rollingLoss = loss;
		else rollingLoss = (rollingLoss * (logInterval - 1) + loss) / logInterval;

		if (i % logInterval == 0)
		{
			printf("[step %d] loss: %.4g\n", i, rollingLoss);
		}
	}

	for (size_t b = 0; b < batchSize; b++)
	{
		delete[] batchInfo[b];
	}
	delete[] batchInfo;
}