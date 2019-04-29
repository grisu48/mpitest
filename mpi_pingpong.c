
/* =============================================================================
 * 
 * Title:       OpenMPI latency test program
 * Author:      Felix Niederwanger
 * 
 * =============================================================================
 */
 
 
 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <mpi.h>
#include <sys/time.h>



static void check_mpi_error(const int rc, const char* func) {
	if(rc != 0) {
		fprintf(stderr, "Error in %s: %s", func, strerror(errno));
	}
}

static void cleanup() {
	MPI_Finalize();
}


int main(int argc, char** argv) {
    long iterations = 1000L;
	check_mpi_error(MPI_Init(&argc, &argv), "MPI_Init");
	atexit(cleanup);

	if(argc > 1) {
		if(!strcmp("-h", argv[1]) || !strcmp("-h", argv[1])) {
			printf("MPI PingPong\n");
			printf("2019 Felix Niederwanger\n");
			printf("Usage: %s [ITERATIONS]\n", argv[0]);
			exit(EXIT_SUCCESS);
		}

		iterations = atol(argv[1]);
	}
	
    int world_size = 0;
    int rank = 0;
    check_mpi_error(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");
    check_mpi_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    if( (world_size % 2) != 0) {
    	if(rank == 0) 
    		fprintf(stderr, "Error: Number of processes must be even\n");
    	exit(EXIT_SUCCESS);
    }
    
    
	struct timeval tv1, tv2, tv_delta;
	int buf = 0;

	if(rank == 0) printf("PingPong with %d ranks and %ld iterations\n", world_size, iterations);
	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&tv1, NULL);		// Set timer 1
    for(long i=0;i<iterations;i++) {
    	if( (rank % 2) == 0) {
			int rank_dest = (rank+1)%world_size;
			check_mpi_error(MPI_Send(&buf, 1, MPI_INT, rank_dest, 1, MPI_COMM_WORLD), "MPI_Send");
		} else {
			// Upper ranks receive from lower ranks
			MPI_Status status;
			int rank_src = (rank-1);
			check_mpi_error(MPI_Recv(&buf, 1, MPI_INT, rank_src, 1, MPI_COMM_WORLD, &status), "MPI_Recv");
		}
    }
    gettimeofday(&tv2, NULL);		// Set timer 2

    timersub(&tv2, &tv1, &tv_delta);	// Delta time
    if(rank == 0) {
    	double millis = (tv_delta.tv_sec*1000.0) + (tv_delta.tv_usec/1000.0);;
    	printf("Runtime: %.2f ms\n", millis);
    }    

    return EXIT_SUCCESS;
}

