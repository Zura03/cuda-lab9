#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void matrix_multiply(int* a, int* b, int* c, int wa) {
	int ridA = threadIdx.y;
	int cidB = threadIdx.x;
	int wb = blockDim.x;

	int sum = 0;
	for (int k = 0; k < wa; k++) {
		sum += a[ridA * wa + k] * b[k * wb + cidB];
	}
	c[ridA * wb + cidB] = sum;
}

int main() {
	
	int* a, * b, * c, ha, wa, hb, wb;
	int* d_a, * d_b, * d_c;

	printf("Enter no.of rows and columns of A: ");
	scanf("%d %d", &ha, &wa);

	printf("Enter no.of rows and columns of B: ");
	scanf("%d %d", &hb, &wb);

	int sizeA = sizeof(int) * ha * wa;
	int sizeB = sizeof(int) * hb * wb;
	int sizeC = sizeof(int) * ha * wb;

	a = (int*)malloc(sizeA);
	b = (int*)malloc(sizeB);
	c = (int*)malloc(sizeC);

	printf("Enter A: ");
	for (int i = 0; i < ha; i++)
		for (int j = 0; j < wa; j++)
			scanf("%d", &a[i * wa + j]);

	printf("Enter B: ");
	for (int i = 0; i < hb; i++)
		for (int j = 0; j < wb; j++)
			scanf("%d", &b[i * wb + j]);

	cudaMalloc((void**)&d_a, sizeA);
	cudaMalloc((void**)&d_b, sizeB);
	cudaMalloc((void**)&d_c, sizeC);

	cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeC, cudaMemcpyHostToDevice);

	matrix_multiply << <dim3(1, 1), dim3(wb, ha) >> > (d_a, d_b, d_c, wa);

	cudaMemcpy(c, d_c, sizeC, cudaMemcpyDeviceToHost);

	printf("C:\n");
	for (int i = 0; i < ha; i++) {
		for (int j = 0; j < wb; j++) {
			printf("%d\t", c[i * wb + j]);
		}
		printf("\n");
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}

