
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

static time_t pingpong(const size_t buf_size, const long iterations) {
	int *buf = (int*)malloc(sizeof(int)*buf_size);
	struct timeval tv1, tv2, tv_delta;
	bzero(buf, sizeof(int)*buf_size);

    int world_size = 0;
    int rank = 0;
    check_mpi_error(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");
    check_mpi_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");

	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&tv1, NULL);		// Set timer 1
    for(long i=0;i<iterations;i++) {
    	if( (rank % 2) == 0) {
			int rank_dest = (rank+1)%world_size;
			check_mpi_error(MPI_Send(buf, buf_size, MPI_INT, rank_dest, 1, MPI_COMM_WORLD), "MPI_Send");
		} else {
			// Upper ranks receive from lower ranks
			MPI_Status status;
			int rank_src = (rank-1);
			check_mpi_error(MPI_Recv(buf, buf_size, MPI_INT, rank_src, 1, MPI_COMM_WORLD, &status), "MPI_Recv");
		}
    }
    gettimeofday(&tv2, NULL);		// Set timer 2

    timersub(&tv2, &tv1, &tv_delta);	// Delta time
   	return (tv_delta.tv_sec*1000L*1000L) + (tv_delta.tv_usec);;
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
    
    
	if(rank == 0) printf("PingPong with %d ranks and %ld iterations\n", world_size, iterations);
	if(rank == 0) printf("#   Size [B]    Runtime [us]\n");
	time_t runtime = 0;
	for(int i=0;i<=14;i++) {
		size_t base_size = (size_t)pow(2,i);
		
		for(int j=0;j<=5;j++) {
			size_t size = base_size + j;
			runtime = pingpong(size, iterations);
			if(rank == 0) {
				printf("%8ld    %8ld\n", size, runtime);
			}
		}
	}
	
	

    return EXIT_SUCCESS;
}

