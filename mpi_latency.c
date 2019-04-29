
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


static void check_mpi_error(const int rc, const char* func) {
	if(rc != 0) {
		fprintf(stderr, "Error in %s: %s", func, strerror(errno));
	}
}

static inline double sum_formula(const long n) { return n*(n+1.0)/2.0; }

static double sum_array(const double *arr, const size_t n) {
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
static double arr_avg(const double *arr, const size_t n) {
	return sum_array(arr, n)/n;
}


static void cleanup() {
	MPI_Finalize();
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
    
    if(verbose) printf("I am rank %d/%d\n", rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    
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
		printf("Rank %d    sum(send_buf) = %lf, Sum(recv_buf) = %lf \n", rank, sum_array(sb, buf_size), sum_array(rb, buf_size));
		MPI_Barrier(MPI_COMM_WORLD);
	}

	struct timeval tv1, tv2, tv_delta;

	if(verbose && rank == 0) printf("Sending buffer (%ld elements) to all nodes ... \n", buf_size);
	double runtime_ms = 0L;
	for(int i=0;i<iterations;i++) {
		gettimeofday(&tv1, NULL);		// Set timer 1
		check_mpi_error(MPI_Allreduce(sb, rb, buf_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce");
		gettimeofday(&tv2, NULL);		// Set timer 2
		if(verbose) printf("MPI_Allreduce (rank %d) ... OK\n", rank);
		if(verbose) {
			MPI_Barrier(MPI_COMM_WORLD);
			printf("Rank %d    sum(send_buf) = %lf, Sum(recv_buf) = %lf \n", rank, sum_array(sb, buf_size), sum_array(rb, buf_size));
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
	    	printf("  Iteration %d: t = %.2f ms\n", i, millis);
	    	runtime_ms += millis;
	    	runtime[i] = millis;
	    }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
    	const long millis = (long)runtime_ms;
    	printf("Total runtime: %ld ms\n", millis);
    	printf("  Worst      : %f ms\n", arr_max(runtime, iterations));
    	printf("  Best       : %f ms\n", arr_min(runtime, iterations));
    	printf("  Average    : %f ms\n", arr_avg(runtime, iterations));
    }


    free(sb);
    free(rb);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) 
    	printf("Bye\n");
    return EXIT_SUCCESS;
}

