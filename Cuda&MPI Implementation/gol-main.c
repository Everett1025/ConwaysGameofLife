#include <stdio.h>
#include<mpi.h>
#include <stdlib.h>
#include<unistd.h>

extern void gol_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank, int numranks);
extern void gol_swap(unsigned char **pA, unsigned char **pB);
extern void gol_printWorld(unsigned char* d_data);
extern void cudaDeviceSynchronize();
extern void launcher(size_t blockCount, unsigned int threadsCount, unsigned char **g_data, unsigned int worldSize, unsigned char **g_resultData);




extern unsigned char* g_data;
extern unsigned char* g_resultData;

int modulo(int x, int N)
{
	return (x % N + N) % N;
}


int main(int argc, char** argv){

	// Initialize the MPI environment
	MPI_Init(&argc, &argv);
	
	int numranks, myrank;

	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);

	// Get the rank of the process
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	double startTime;

	if(myrank == 0)
	{
		startTime = MPI_Wtime();
	}

	unsigned int pattern = atoi(argv[1]);
	unsigned int worldSize = atoi(argv[2]);
	unsigned int itterations = atoi(argv[3]);
	unsigned int threadsCount = atoi(argv[4]);
	int output = atoi(argv[5]);



	gol_initMaster(pattern, worldSize, worldSize + 2, myrank, numranks);
	size_t blockCount = (worldSize * (worldSize + 2) / threadsCount);

		
	for(int i = 0; i < itterations; i ++)
	{	
		
		MPI_Request reqs[4];
		MPI_Status stats[4];

		MPI_Irecv(g_data, worldSize, MPI_INT, modulo((myrank - 1), numranks), 2, MPI_COMM_WORLD, &reqs[0]);	
		MPI_Irecv(g_data + (worldSize * (worldSize + 1)), worldSize, MPI_INT,  modulo((myrank + 1), numranks), 2, MPI_COMM_WORLD, &reqs[1]);
	
		MPI_Isend(g_data + (worldSize * (worldSize + 0)), worldSize, MPI_INT, modulo((myrank + 1), numranks), 2, MPI_COMM_WORLD, &reqs[3]);
		MPI_Isend(g_data + worldSize, worldSize, MPI_INT, modulo((myrank - 1), numranks), 2, MPI_COMM_WORLD, &reqs[2]);
	
		MPI_Waitall(4, reqs, stats);
		launcher(blockCount, threadsCount, &g_data, worldSize, &g_resultData);			
	}

	cudaDeviceSynchronize();
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank == 0)
	{
		printf("Here is the wait time %f", MPI_Wtime() - startTime);
	}
	
	if(output){
		for(int i = 0; i < numranks; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if( i == myrank)
			{		
				printf("This is rank %d.\n", myrank);
				printf("######################### FINAL WORLD IS ###############################\n");
				gol_printWorld(g_data);		
				fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
	MPI_Finalize();

	return 0;
}
