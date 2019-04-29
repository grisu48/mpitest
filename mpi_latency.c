
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


static bool verbose = false;
static size_t buf_size = 40*1024L;
static int iterations = 5;



static inline double sum_formula(const long n) { return n*(n+1.0)/2.0; }
static double arr_sum(const double *arr, const size_t n) {
	double sum = 0.0;
	for(size_t i=0;i<n;i++)
		sum += arr[i];
	return sum;
}
static double arr_max(const double *arr, const size_t n) {
	if(n <= 0) return 0;
	double ret = arr[0];
	for(size_t i=0;i<n;i++)
		ret = fmax(ret, arr[i]);
	return ret;
}
static double arr_min(const double *arr, const size_t n) {
	if(n <= 0) return 0;
	double ret = arr[0];
	for(size_t i=0;i<n;i++)
		ret = fmin(ret, arr[i]);
	return ret;
}





static void check_mpi_error(const int rc, const char* func) {
	if(rc != 0) {
		fprintf(stderr, "Error in %s: %s", func, strerror(errno));
	}
}

static void cleanup() {
	MPI_Finalize();
}

typedef struct {
	double total;
	double worst;
	double best;
	double avg;
} runstat_t;
static void clear_stat(runstat_t *stat) {
	stat->total = 0;
	stat->worst = 0;
	stat->best = 0;
	stat->avg = 0;
}


static void parse_args(const int argc, const char** argv) {
	int param = 0;
	for(int i=1;i<argc;i++) {
		const char* arg = argv[i];
		if(!strcmp("-h", arg) || !strcmp("--help", arg)) {
			printf("Simple MPI latency test program\n");
			printf("2019 Felix Niederwanger\n");
			printf("\n");
			printf("Usage: %s -h, --help            Print this help message\n", argv[0]);
			printf("Usage: %s [-v] [n] [N]\n", argv[0]);
			printf("  -v    Verbose run\n");
			printf("  n ... Array size (in kB)\n");
			printf("  N ... Number of iterations\n");
			exit(EXIT_SUCCESS);
		} else if(!strcmp("-v", arg) || !strcmp("--verbose", arg)) {
			verbose = true;
		} else {
			switch(param++) {
				case 0:
					buf_size = atol(arg) * 1024L;
					break;
				case 1:
					iterations = atoi(arg);
					break;
			}
		}
	}

	if(buf_size <= 0) {
		fprintf(stderr, "Illegal buffer size: %ld\n", buf_size);
		exit(EXIT_FAILURE);
	}
	if(iterations <= 0) {
		fprintf(stderr, "Illegal iterations: %d\n", iterations);
		exit(EXIT_FAILURE);
	}
}


int main(int argc, char** argv) {
	check_mpi_error(MPI_Init(&argc, &argv), "MPI_Init");
	atexit(cleanup);

	parse_args(argc, (const char**)argv);
	
    int world_size = 0;
    int rank = 0;
    check_mpi_error(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");
    check_mpi_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    if( (world_size % 2) != 0) {
    	if(rank == 0) 
    		fprintf(stderr, "Error: Number of processes must be even\n");
    	exit(EXIT_SUCCESS);
    }
    
    if(verbose) printf("I am rank %d/%d\n", rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stats
    runstat_t stat_all, stat_neighbour;
    clear_stat(&stat_all);
    clear_stat(&stat_neighbour);
    
    
	double *sb; // Send Buffer
	double *rb; // Receive buffer
	double *runtime;	// Runtimes

	sb = (double*)malloc(buf_size * sizeof(double));
	rb = (double*)malloc(buf_size * sizeof(double));
	runtime = (double*)malloc(iterations * sizeof(double));
	if(sb == NULL || rb == NULL || runtime == NULL) {
		fprintf(stderr, "rank %d - failed to initialize buffers (out of memory)\n", rank);
		exit(EXIT_FAILURE);
	}

	for(size_t i=0; i<buf_size;i++) {
		sb[i] = 1.0;
		rb[i] = 0;
	}

	if(verbose) {
		printf("Rank %d    sum(send_buf) = %lf, Sum(recv_buf) = %lf \n", rank, arr_sum(sb, buf_size), arr_sum(rb, buf_size));
		MPI_Barrier(MPI_COMM_WORLD);
	}

	struct timeval tv1, tv2, tv_delta;

	if(rank == 0) printf("ALL-ALL Test: Sending buffer (%ld elements) to all nodes ... \n", buf_size);
	for(int i=0;i<iterations;i++) {
		gettimeofday(&tv1, NULL);		// Set timer 1
		check_mpi_error(MPI_Allreduce(sb, rb, buf_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce");
		gettimeofday(&tv2, NULL);		// Set timer 2
		if(verbose) printf("MPI_Allreduce (rank %d) ... OK\n", rank);
		if(verbose) {
			MPI_Barrier(MPI_COMM_WORLD);
			printf("Rank %d    sum(send_buf) = %lf, Sum(recv_buf) = %lf \n", rank, arr_sum(sb, buf_size), arr_sum(rb, buf_size));
			MPI_Barrier(MPI_COMM_WORLD);
		}
	    
		// Check result for consistency
		const double exp_ret = world_size;
		for(size_t i=0; i<buf_size;i++) {
			if(rb[i] != exp_ret) {
				fprintf(stderr, "rb :: %lf != %lf\n", rb[i], exp_ret);
				exit(EXIT_FAILURE);
			}
		}

	    timersub(&tv2, &tv1, &tv_delta);	// Delta time
	    if(rank == 0) {
	    	double millis = (tv_delta.tv_sec*1000.0) + (tv_delta.tv_usec/1000.0);;
	    	printf("  ALL-ALL: Iteration %d: t = %.2f ms\n", i, millis);
	    	runtime[i] = millis;
	    }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
    	stat_all.total = arr_sum(runtime, iterations);
    	stat_all.worst = arr_max(runtime, iterations);
    	stat_all.best = arr_min(runtime, iterations);
    	stat_all.avg = stat_all.total/iterations;
    	printf("Total runtime: %f ms\n", stat_all.total);
    	printf("  Worst      : %f ms\n", stat_all.worst);
    	printf("  Best       : %f ms\n", stat_all.best);
    	printf("  Average    : %f ms\n", stat_all.avg);
    }

	// Test two: Neigbour test
	if(rank == 0) printf("ALL-ALL Test: Sending buffer (%ld elements) to all nodes ... \n", buf_size);
	for(int i=0;i<iterations;i++) {
		
		const int tag = i;
		gettimeofday(&tv1, NULL);		// Set timer 1
		if( (rank % 2) == 0) {
			int rank_dest = (rank+1)%world_size;
			check_mpi_error(MPI_Send(sb, buf_size, MPI_DOUBLE, rank_dest, tag, MPI_COMM_WORLD), "MPI_Send");
		} else {
			// Upper ranks receive from lower ranks
			MPI_Status status;
			int rank_src = (rank-1);
			check_mpi_error(MPI_Recv(rb, buf_size, MPI_DOUBLE, rank_src, tag, MPI_COMM_WORLD, &status), "MPI_Recv");
		}
		MPI_Barrier(MPI_COMM_WORLD);
		gettimeofday(&tv2, NULL);		// Set timer 2
		
	    timersub(&tv2, &tv1, &tv_delta);	// Delta time
	    if(rank == 0) {
	    	double millis = (tv_delta.tv_sec*1000.0) + (tv_delta.tv_usec/1000.0);;
	    	printf("  Neighbour: Iteration %d: t = %.2f ms\n", i, millis);
	    	runtime[i] = millis;
	    }
	}
	
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
    	stat_neighbour.total = arr_sum(runtime, iterations);
    	stat_neighbour.worst = arr_max(runtime, iterations);
    	stat_neighbour.best = arr_min(runtime, iterations);
    	stat_neighbour.avg = stat_neighbour.total/iterations;
    	printf("Total runtime: %f ms\n", stat_neighbour.total);
    	printf("  Worst      : %f ms\n", stat_neighbour.worst);
    	printf("  Best       : %f ms\n", stat_neighbour.best);
    	printf("  Average    : %f ms\n", stat_neighbour.avg);
    }
	
    free(sb);
    free(rb);
    MPI_Barrier(MPI_COMM_WORLD);
    
	// Print summary
	if(rank == 0) {
		printf("\n\n");
		printf("================================================================================\n");
		printf("  SUMMARY\n");
		printf("  World size   : %d\n", world_size);
		printf("  Buffer size  : %ld\n", buf_size);
		printf("  Iterations   : %d\n", iterations);
		printf("\n");
		printf("Global test (MPI_Allreduce)\n");
		printf("  Average: %6.4f ms    Worst: %6.4f    Best: %6.4f\n", stat_all.avg, stat_all.worst, stat_all.best);
		printf("\n");
		printf("Neighbour test (MPI_Send, MPI_Recv)\n");
		printf("  Average: %6.4f ms    Worst: %6.4f    Best: %6.4f\n", stat_neighbour.avg, stat_neighbour.worst, stat_neighbour.best);
		printf("\n");
		printf("================================================================================\n");
	}

    return EXIT_SUCCESS;
}

