#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <iomanip>

#define SIZE 1024

// Matrix is a 4x4 array of array of ints
#define ORDER 3
#define MAT_SIZE ORDER*ORDER

#define MATRIX_COMP_MUL(_i, _j, _MATA, _MATB, _DEST) \
	_DEST[_i * ORDER + _j] = 0; \
	for(size_t k = 0; k < ORDER; ++k) \
		_DEST[_i * ORDER + _j] += _MATA[_i * ORDER + k] * _MATB[k * ORDER + _j];

#define PRINT_MATRIX_ROW(_row) \
	std::cout << "| "; \
	for(size_t _i = 0; _i < ORDER; ++_i) \
		std::cout << std::setw(5) << std::right << (_row)[_i] << ' '; \
	std::cout << '|';

__global__ void matrixMultiplyKernel(const int* matAPtr, const int* matBPtr, int* matDestPtr) {
	int id = threadIdx.x * MAT_SIZE;

	const int* matA = &matAPtr[id];
	const int* matB = &matBPtr[id];
    int* matDest = &matDestPtr[id];

	for (size_t i = 0; i < ORDER; ++i) {
		for (size_t j = 0; j < ORDER; ++j) {
			MATRIX_COMP_MUL(i, j, matA, matB, matDest);
		}
	}
}

/* Generates a random matrix. */
void generateRandomMatrix(int* mat) {
	static std::default_random_engine engine;
	std::uniform_int_distribution<int> dist(1, 200);
	for (size_t i = 0; i < ORDER; ++i)
		for (size_t j = 0; j < ORDER; ++j)
			mat[i * ORDER + j] = dist(engine);
}

/* Generates a list of N Random Matrices. Value of N is the value of SIZE. */
static void generateRandomMatrices(int* dest) {
	for (size_t i = 0; i < SIZE; ++i) {
		generateRandomMatrix(&dest[i * MAT_SIZE]);
	}
}

// Pointers to the list of matrices in GPU.
int* matricesAGPU;
int* matricesBGPU;
int* matricesDestGPU;

void GPUCleanup() {
	cudaFree(matricesAGPU);
	cudaFree(matricesBGPU);
	cudaFree(matricesDestGPU);
}

/* Multiply the given matrices in parallel using CUDA. The source and destination arrays must have atleast
   MAT_ORDER * MAT_ORDER * SIZE number of elements.
   Arguments:                                         
   a_mats: Pointer to the first element of a 3D array containing the values of the first matrix operands.
   b_mats: Pointer to the first element of a 3D array containing the values of the second matrix operands.
   dest_mats: Pointer to the first element of a 3D array containing the values of the result matrices.
*/
cudaError_t matrixMultiplyWithCuda(const int * matricesA, const int * matricesB, int * matricesDest) {
	cudaError_t status;
	status = cudaSetDevice(0);
	if (status != cudaSuccess) {
		std::cerr << "Error initializing CUDA! Please check your GPU!" << std::endl;
		return status;
	}

	constexpr int sz = ORDER * ORDER * SIZE;

	std::cout << "Allocating memory on GPU for data...\r";
	status = cudaMalloc((void**)&matricesAGPU, sz * sizeof(int));
	if (status != cudaSuccess) {
		GPUCleanup();
		std::cerr << "Error allocating memory on CUDA device!" << std::endl;
		return status;
	}

	status = cudaMalloc((void**)&matricesBGPU, sz * sizeof(int));
	if (status != cudaSuccess) {
		GPUCleanup();
		std::cerr << "Error allocating memory on CUDA device!" << std::endl;
		return status;
	}

	status = cudaMalloc((void**)&matricesDestGPU, sz * sizeof(int));
	if (status != cudaSuccess) {
		GPUCleanup();
		std::cerr << "Error allocating memory on CUDA device!" << std::endl;
		return status;
	}
	std::cout << "Successfully allocated memory on the GPU device!" << std::endl;

	std::cout << "Copying data to GPU for calculation...\r";
	status = cudaMemcpy(matricesAGPU, matricesA, sz * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		GPUCleanup();
		std::cerr << "Error copying memory from CUDA device!" << std::endl;
		return status;
	}
	status = cudaMemcpy(matricesBGPU, matricesB, sz * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		GPUCleanup();
		std::cerr << "Error copying memory from CUDA device!" << std::endl;
		return status;
	}
	std::cout << "Successfully copied all the data to the GPU device!" << std::endl;

	matrixMultiplyKernel<<<1, SIZE>>>(matricesAGPU, matricesBGPU, matricesDestGPU);

	std::cout << "Cleaning up..." << std::endl;

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		GPUCleanup();
		std::cout << "Error synchronizing GPU! Debugging required." << std::endl;
		return status;
	}

	status = cudaMemcpy((void*)matricesDest, matricesDestGPU, sz * sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		GPUCleanup();
		std::cerr << "Error copying memory from CUDA device!" << std::endl;
		return status;
	}

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		GPUCleanup();
		std::cerr << "Error resetting on GPU! Debugging required." << std::endl;
		return status;
	}

	GPUCleanup();
	return status;
}

void printMatrix(const int* a, const int* b, const int* c) {
	char sep = ' ';
	char eq = ' ';
	for (size_t i = 0; i < ORDER; ++i) {
		if (i + 1 == ORDER / 2) {
			sep = 'x';
			eq = '=';
		}
		else {
			sep = ' ';
			eq = ' ';
		}

		PRINT_MATRIX_ROW(&a[i * ORDER]);
		std::cout << sep;
		PRINT_MATRIX_ROW(&b[i * ORDER]);
		std::cout << eq;
		PRINT_MATRIX_ROW(&c[i * ORDER]);
		std::cout << '\n';
	}
}

void printMatrices(const int* matricesA, const int* matricesB, const int* matricesC) {
	std::ios::sync_with_stdio(false);
	std::cout << std::endl;
	std::cout << std::setprecision(5);
	for (size_t i = 0; i < SIZE; ++i) {
		printMatrix(&matricesA[i * MAT_SIZE], &matricesB[i * MAT_SIZE], &matricesC[i * MAT_SIZE]);
		std::cout << '\n';
	}
	std::cout << std::flush;
	std::ios::sync_with_stdio(true);
}

int main() {
	std::cout << "CUDA Test made by Dot." << std::endl;
	std::cout << "CUDA Program to multiply " << SIZE << " number of " 
		<< ORDER << 'x' << ORDER
		<< " random matrices and print their result." << std::endl;

	int* matricesA = new int[MAT_SIZE * SIZE];
	int* matricesB = new int[MAT_SIZE * SIZE];
	int* matricesDest = new int[MAT_SIZE * SIZE];

	std::cout << "Generating random matrices...\r";
	generateRandomMatrices(matricesA);
	generateRandomMatrices(matricesB);
	std::cout << "Random matrices for use as operands have been generated!" << std::endl;

	std::cout << "Multiplying matrices using CUDA..." << std::endl;
	matrixMultiplyWithCuda(matricesA, matricesB, matricesDest);
	std::cout << "Successfully multiplied the matrices using CUDA! Printing results now..." << std::endl;

	printMatrices(matricesA, matricesB, matricesDest);

	std::cout << "Program successfully executed!" << std::endl;

	return 0;
}
