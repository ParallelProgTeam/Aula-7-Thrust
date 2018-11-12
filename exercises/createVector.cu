#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

#ifndef checkCudaErrors
static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define checkCudaErrors( err ) (HandleError( err, __FILE__, __LINE__ ))	
#endif

int main(void)
{
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));
    float totalTime = 0;


    checkCudaErrors(cudaEventRecord(start_event, 0));


    // Declare AQUI um vetor de host H para armazenar 4 inteiros 
    thrust::host_vector<int> H(4);

    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;
    
    // H.size() returns the size of vector H
    std::cout << "H has size " << H.size() << std::endl;

    // print contents of H
    for(int i = 0; i < H.size(); i++)
        std::cout << "H[" << i << "] = " << H[i] << std::endl;

    // resize H
    H.resize(2);
    
    std::cout << "H now has size " << H.size() << std::endl;

    // Copy host_vector H to device_vector D
    thrust::device_vector<int> D = H;
    
    // elements of D can be modified
    D[0] = 99;
    D[1] = 88;
    

    // print contents of D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors( cudaEventElapsedTime( &totalTime, start_event, stop_event ) );
    std::cout << "Total time: " << totalTime << " ms" << std::endl;

    // H and D are automatically deleted when the function returns
    return 0;
}
