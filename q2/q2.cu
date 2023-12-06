#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#define MASK_WIDTH 3
#define MASK_RADIUS MASK_WIDTH/2

__global__ void convolution2D(float* input, float* output, int width, int height, float* mask) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < height && col < width) {
		float sum = 0.0f;
		for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++) {
			for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
				int r = row + i;
				int c = col + j;
				if (r >= 0 && r < height && c >= 0 && c < width) {
					sum += input[r * width + c] * mask[(i + MASK_RADIUS) * MASK_WIDTH + (j + MASK_RADIUS)];
				}
			}
		}
		output[row * width + col] = sum;
	}
}

int main() {
	int width, height;
	printf("Enter the height and width of the input array: ");
	scanf("%d %d", &height, &width);

	const int inputsize = width * height;
	const int outputsize = inputsize;

	float* hostinput = (float*)malloc(inputsize * sizeof(float));
	float* hostoutput = (float*)malloc(outputsize * sizeof(float));
	float* hostmask = (float*)malloc(MASK_WIDTH * MASK_WIDTH * sizeof(float));

	printf("Enter the values of the input array (%d elements): \n", inputsize);
	for (int i = 0; i < inputsize; i++)
		scanf("%f", &hostinput[i]);

	printf("Enter the values of the mask array (%d elements)", MASK_WIDTH * MASK_WIDTH);
	for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++)
		scanf("%f", &hostmask[i]);

	float* deviceinput, * deviceoutput, * devicemask;

	cudaMalloc((void**)&deviceinput, inputsize * sizeof(float));
	cudaMalloc((void**)&deviceoutput, outputsize * sizeof(float));
	cudaMalloc((void**)&devicemask, MASK_WIDTH*MASK_WIDTH * sizeof(float));

	cudaMemcpy(deviceinput, hostinput, inputsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devicemask, hostmask, MASK_WIDTH*MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	dim3 gridsize((width + 15) / 16, (height + 15) / 16);
	dim3 blocksize(16, 16);

	convolution2D << <gridsize, blocksize >> > (deviceinput, deviceoutput, width, height, devicemask);

	cudaMemcpy(hostoutput, deviceoutput, outputsize * sizeof(float), cudaMemcpyDeviceToHost);

	printf("convolution result: \n");
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%2f\t", hostoutput[i * width + j]);
		}
		printf("\n");
	}

	cudaFree(deviceinput);
	cudaFree(deviceoutput);
	cudaFree(devicemask);

	return 0;
}
