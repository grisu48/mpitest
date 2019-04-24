
/* =============================================================================
 * 
 * Title:       Simple OpenMPI test program
 * Author:      Felix Niederwanger
 * 
 * =============================================================================
 */
 
 
 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <mpi.h>


#define BUF_SIZE 2048


static void check_mpi_error(const int rc, const char* func) {
	if(rc != 0) {
		fprintf(stderr, "Error in %s: %s", func, strerror(errno));
	}
}

static void cleanup() {
	MPI_Finalize();
}

/**
  * Set data in buffer to rank data
  */
static void set_data(float *buffer, const size_t size, const int rank) {
    for(size_t i=0; i<size; i++) buffer[i] = rank;
}

/**
  * Checks if the given data matches produced data from the given rank
  */
static bool check_data(const float *buffer, const size_t size, const int rank) {
	for(size_t i=0; i<size; i++) 
		if (buffer[i] != rank) return false;
	return true;
}


int main(int argc, char** argv) {
	check_mpi_error(MPI_Init(&argc, &argv), "MPI_Init");
	atexit(cleanup);
	
    int world_size = 0;
    int rank = 0;
    check_mpi_error(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");
    check_mpi_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    if( (world_size % 2) != 0) {
    	if(rank == 0) 
    		fprintf(stderr, "Error: Number of processes must be even\n");
    	exit(EXIT_SUCCESS);
    }
    //printf("I am rank %d/%d\n", rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    	printf("Running MPI tests ... \n");
    
    // Allocate send and receive buffers
    float buffer[BUF_SIZE];
    bzero(buffer, BUF_SIZE*sizeof(float));

    // Keep in mind, MPI_Send returns when the message has been received on the other end.
    // So we always need a sender and a receiver at the same time.
    
    const int tag = 1;

    // Neighbour send test
    if( (rank % 2) == 0) {
    	set_data(buffer, BUF_SIZE, rank);
    	int rank_dest = (rank+1)%world_size;
	    check_mpi_error(MPI_Send(buffer, BUF_SIZE, MPI_FLOAT, rank_dest, tag, MPI_COMM_WORLD), "MPI_Send");
	} else {
    	// Upper ranks receive from lower ranks
    	MPI_Status status;
    	int rank_src = (rank-1);
    	check_mpi_error(MPI_Recv(buffer, BUF_SIZE, MPI_FLOAT, rank_src, tag, MPI_COMM_WORLD, &status), "MPI_Recv");
    	
    	if(!check_data(buffer, BUF_SIZE, rank_src)) {
    		fprintf(stderr, "Error: Rank %d received illegal data\n", rank);
    		exit(EXIT_FAILURE);
    	}
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) 
    	printf("- Neighbour test completed\n");

    // Second test: Every rank sends to every other rank
    for(int i=0;i<world_size;i++) {
    	if(rank == i) { // Is it me to send it to all?
    		set_data(buffer, BUF_SIZE, rank);
    		for(int dest=0;dest<world_size;dest++) {
    			if(dest == rank) continue;
    			else
    				check_mpi_error(MPI_Send(buffer, BUF_SIZE, MPI_FLOAT, dest, tag, MPI_COMM_WORLD), "MPI_Send");
    		}
    	} else {
    		// Await data
    		MPI_Status status;
    		check_mpi_error(MPI_Recv(buffer, BUF_SIZE, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status), "MPI_Recv");

    	if(!check_data(buffer, BUF_SIZE, i)) {
    		fprintf(stderr, "Error: Rank %d received illegal data\n", rank);
    		exit(EXIT_FAILURE);
    	}
    	}
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) 
    	printf("- All-to-All test completed\n");

    if(rank == 0) 
    	printf("All good\n");
    return EXIT_SUCCESS;
}

