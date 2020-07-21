#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

// THE FOLLOWING CODE IS A C IMPLEMENTATION ON CONWAYS GAME OF LIFE
// USING PARALLEL IMPLEMENTATION.
// MAJORITY OF THE CODE WAS PROVIDED BY THE PROFESSOR
// TWO KERNEL FUNCTIONS WERE ADDED BY ME TO ASSIST WITH THE PARALLEL IMPLEMENTATION
// THE SWAP POINTER FUNCTION WAS ALSO MY IMPLEMENTATION FROM THE PREVIOUS HOMEWORK 1 ASSIGNMENT

// Result from last compute of world.
unsigned char *g_resultData = NULL;

// Current state of world.
unsigned char *g_data = NULL;

// Current width of world.
size_t g_worldWidth = 0;

/// Current height of world.
size_t g_worldHeight = 0;

/// `Current data length (product of width and height)
size_t g_dataLength = 0;

// PMODIFIED FOR CUDA IMPLEMENTATION
static inline void gol_initAllZeros(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, g_dataLength*sizeof(unsigned char));
  
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, g_dataLength*sizeof(unsigned char));
}

// MODIFIED FOR CUDA IMPLEMENTATION
static inline void gol_initAllOnes(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));

    // set all rows of world to true
    for (i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 1;
    }
     cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
}

// MODIFIED FOR CUDA IMPLEMENTATION
static inline void gol_initOnesInMiddle(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

     cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));

    // set first 1 rows of world to true
    for (i = 10 * g_worldWidth; i < 11 * g_worldWidth; i++)
    {
        if ((i >= (10 * g_worldWidth + 10)) && (i < (10 * g_worldWidth + 20)))
        {
            g_data[i] = 1;
        }
    }
     cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
}

// MODIFIED FOR CUDA IMPLEMENTATIONE
static inline void gol_initOnesAtCorners(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));

    g_data[0] = 1;                                                 // upper left
    g_data[worldWidth - 1] = 1;                                    // upper right
    g_data[(worldHeight * (worldWidth - 1))] = 1;                  // lower left
    g_data[(worldHeight * (worldWidth - 1)) + worldWidth - 1] = 1; // lower right

    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));

}

// MODIFIED FOR CUDA IMPLEMENTATIONE
static inline void gol_initSpinnerAtCorner(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));

    g_data[0] = 1;              // upper left
    g_data[1] = 1;              // upper left +1
    g_data[worldWidth - 1] = 1; // upper right

    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));

}

// CREATES THE WORLD
static inline void gol_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight)
{
    switch (pattern)
    {
    case 0:
        gol_initAllZeros(worldWidth, worldHeight);
        break;

    case 1:
        gol_initAllOnes(worldWidth, worldHeight);
        break;

    case 2:
        gol_initOnesInMiddle(worldWidth, worldHeight);
        break;

    case 3:
        gol_initOnesAtCorners(worldWidth, worldHeight);
        break;

    case 4:
        gol_initSpinnerAtCorner(worldWidth, worldHeight);
        break;

    default:
        printf("Pattern %u has not been implemented \n", pattern);
        exit(-1);
    }
}


// SWAPPING THE VALUES OF TWO POINTER VARIABLES "PA" AND "PB"
// WE DO THIS BY PASSING IN THE MEMORY OF ADDRESS OF EACH PARAMETER
static inline void gol_swap(unsigned char **pA, unsigned char **pB)
{
    unsigned char * temp = *pA;
    *pA = *pB;
    *pB = temp;
    return;
}



// PRINTS WORLD USING THE GLOBAL VARIABLE
static inline void gol_printWorld()
{
    int i, j;
    for (i = 0; i < g_worldHeight; i++)
    {
        printf("Row %2d: ", i);
        for (j = 0; j < g_worldWidth; j++)
        {
            printf("%u ", (unsigned int)g_data[(i * g_worldWidth) + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}


// MAIN CUDA FUNCTION THAT CALCULATES IF A CELL SHOULD BE ALIVE OR DEAD
__global__ void gol_kernel(const unsigned char* d_data, unsigned int worldWidth, unsigned int worldHeight, unsigned char* d_resultData){

    // COMPUTE THE SIZE OF OUR 1D ARRAY WHCIH REPRESENTS OUR WORLD
    unsigned int worldSize = worldWidth * worldHeight; 

    // LOOP THROUGH OUR WORLD
    for (unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x; index < worldSize; index += blockDim.x * gridDim.x){

	// DETERMINE LOCATIONS OF SURROUNDING CELLS
        size_t x = index % worldWidth;
        size_t y = index - x;
        size_t yUp = ( y + worldSize - worldWidth) % worldSize;
	size_t yDown = (y + worldWidth) % worldSize;
	size_t xLeft = ( x + worldWidth - 1) % worldWidth;
        size_t xRight = ( x + 1) % worldWidth;
        
        // DETERMINE COUNT OF SORROUNDING CELLS THAT ARE ALIVE
	unsigned int aliveCount = d_data[yUp + xLeft] + d_data[y + xLeft] + d_data[yDown + xLeft] + d_data[yUp + x] + d_data[yDown + x] + d_data[yUp + xRight] + d_data[y + xRight] + d_data[yDown + xRight];

        d_resultData[x + y] = aliveCount == 3 || (aliveCount == 2 && d_data[x + y]) ? 1 : 0;
    }
}




// COMPUTES WORLD VIA CUDA KERNEL AND SWAPS THE NEW WORLD WITH THE PERVIOUS
// INVOKES THE FUNCTION gol_kernel
void gol_kernelLanuch(unsigned char **d_data, unsigned char** d_resultData, size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount){

    size_t blockCount = (worldWidth * worldHeight) / threadsCount;

    for(size_t i = 0; i < iterationsCount; ++i){
        gol_kernel<<<blockCount, threadsCount>>>(*d_data, worldWidth, worldHeight, *d_resultData);
	gol_swap(d_resultData, d_data);
    }
	 cudaDeviceSynchronize();
}


// MAIN
// CODE HAS BEEN MODIFIED
// ADDED FUNCTION gol_kernelLaunch TO IMPLEMENT PARALLEL VERSION
int main(int argc, char *argv[])
{
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int itterations = 0;
    unsigned int threadsCount = 0;
    int output;
    printf("This is the Game of Life running in serial on a CPU.\n");

    if (argc != 6)
    {
        printf("GOL requires 5 arguments: pattern number, sq size of the world and the number of itterations, thread count, and toggled output, e.g. ./gol 0 32 2 2 0 \n");
        exit(-1);
    }
    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    itterations = atoi(argv[3]);
    threadsCount = atoi(argv[4]);
    output = atoi(argv[5]);
    gol_initMaster(pattern, worldSize, worldSize);

    gol_kernelLanuch(&g_data, &g_resultData, worldSize, worldSize, itterations, threadsCount);

    if(output){
        printf("######################### FINAL WORLD IS ###############################\n");
        gol_printWorld();
    }

    cudaFree(g_data);
    cudaFree(g_resultData);

    return true;
}
