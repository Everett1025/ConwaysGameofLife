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
unsigned char *g_resultData;

// Current state of world.
unsigned char *g_data;

// Current width of world.
size_t g_worldWidth = 0;

/// Current height of world.
size_t g_worldHeight = 0;

/// `Current data length (product of width and height)
size_t g_dataLength = 0;

// MODIFIED FOR CUDA IMPLEMENTATION
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
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
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
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
	
    
    // set all rows of world to true
    for (int i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 0;
    }
     cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    
    for (int i = 0; i < 10; i++)
    {
	*(g_data + (worldWidth * (worldHeight - 2)) + 127 + i ) = 1;

    }
 
}

// MODIFIED FOR CUDA IMPLEMENTATIONE
static inline void gol_initOnesAtCorners(size_t worldWidth, size_t worldHeight, int myrank, int numranks)
{

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, g_dataLength*sizeof(unsigned char));
  
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, g_dataLength*sizeof(unsigned char));
   
    if(myrank == 0)
    {
	for(int i = 0; i < worldWidth; i++)
	{
		*(g_data + worldWidth + i) = 1;
	}
    }

    if(myrank == numranks - 1)
    {
	for(int i = 0; i < worldWidth; i++)
	{
		*(g_data + (g_worldWidth * (g_worldHeight - 2) + i )) = 1;    
	}

    }
}

// MODIFIED FOR CUDA IMPLEMENTATIONE
static inline void gol_initSpinnerAtCorner(size_t worldWidth, size_t worldHeight, int myrank)
{
    	g_worldWidth = worldWidth;
    	g_worldHeight = worldHeight;
    	g_dataLength = g_worldWidth * g_worldHeight;

    	cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
	

	if (myrank == 0)
	{
    	g_data[worldWidth] = 1;              // upper left
    	g_data[worldWidth + 1] = 1;              // upper left +1
    	g_data[worldWidth + worldWidth- 1] = 1; // upper right
	}
    	cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
}



static inline void gol_initMyTest(size_t worldWidth, size_t worldHeight, int myrank, int numranks)
{

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, g_dataLength*sizeof(unsigned char));
  
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, g_dataLength*sizeof(unsigned char));
   
    
     
    // set all rows of world to true
    for (int i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 0;
    }
     cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    
    for (int i = 0; i < 10; i++)
    {
	*(g_data + (worldWidth * (worldHeight/2)) + 5 + i ) = 1;

    }

}

static inline void gol_initMyTest2(size_t worldWidth, size_t worldHeight, int myrank, int numranks)
{

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, g_dataLength*sizeof(unsigned char));
  
    cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, g_dataLength*sizeof(unsigned char));
   
    
     
    // set all rows of world to true
    for (int i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 0;
    }
     cudaMallocManaged(&g_resultData, (g_dataLength * sizeof(unsigned char)));
    
    for (int i = 0; i < 10; i++)
    {
	*(g_data + (worldWidth + i)) = 1;	

    }
}



extern "C" void gol_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank, int numranks)
{
    int cudaDeviceCount;
    int cE;

    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess)
    {

	printf("Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
	exit(-1);

    }
    if ( (cE = cudaSetDevice( myrank % cudaDeviceCount)) != cudaSuccess)
    {	
	printf("Unable to have rank %d set to cuda device %d, error is %d \n", myrank, (myrank % cudaDeviceCount), cE);
	exit(-1);
    }
		
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
        gol_initOnesAtCorners(worldWidth, worldHeight, myrank, numranks);
        break;

    case 4:
	gol_initSpinnerAtCorner(worldWidth, worldHeight, myrank);
	break;
    case 8:
	gol_initMyTest(worldWidth, worldHeight, myrank, numranks);
        break;
    case 9:
	gol_initMyTest2(worldWidth, worldHeight, myrank, numranks);
	break;
    default:
        printf("Pattern %u has not been implemented \n", pattern);
        exit(-1);
    }
}



// SWAPPING THE VALUES OF TWO POINTER VARIABLES "PA" AND "PB"
// WE DO THIS BY PASSING IN THE MEMORY OF ADDRESS OF EACH PARAMETER
extern "C" void gol_swap(unsigned char **pA, unsigned char **pB)
{
    unsigned char * temp = *pA;
    *pA = *pB;
    *pB = temp;
    return;
}


// PROVIDED CODE
// PRINTS WORLD USING THE GLOBAL VARIABLE
extern "C" void gol_printWorld(char *d_data)
{
  
    int i, j;
    for (i = 1; i < g_worldHeight-1; i++)
    {
	
        printf("Row %2d: ", i );
        for (j = 0; j < g_worldWidth; j++)
        {
            printf("%u ", (unsigned int)d_data[(i * g_worldWidth) + j]);
        }
        printf("\n");
	fflush(stdout);
    }
    printf("\n\n");

}


// MAIN CUDA FUNCTION THAT CALCULATES IF A CELL SHOULD BE ALIVE OR DEAD
void __global__ gol_kernel(const unsigned char* d_data, unsigned int worldWidth, unsigned int worldHeight, unsigned char* d_resultData){

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

void gol_kernelLaunch(unsigned char **d_data, unsigned char ** d_resultData, size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount)
{
	size_t blockCount = (worldWidth * worldHeight) / threadsCount;
	
	for(size_t i = 0; i < iterationsCount; ++i){
		gol_kernel<<<blockCount, threadsCount>>>(*d_data, worldWidth, worldHeight + 2, *d_resultData);
		gol_swap(d_resultData, d_data);
	}
	cudaDeviceSynchronize();



}

extern "C" void launcher(size_t blockCount, unsigned int threadsCount, unsigned char **d_data, unsigned int worldSize, unsigned char **d_resultData){

	      	gol_kernel<<<blockCount, threadsCount>>>( *d_data, worldSize, worldSize + 2, *d_resultData);
		cudaDeviceSynchronize();
		gol_swap(d_resultData, d_data);
}
