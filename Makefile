

# Default compiler and compiler flags
MPICC=mpicc

# Default flags for all compilers
O_FLAGS=-O3 -Wall -Werror -Wextra -pedantic
# Debugging flags
#O_FLAGS=-Og -g2 -Wall -Werror -Wextra -pedantic
CC_FLAGS=$(O_FLAGS) -std=c99


# Binaries, object files, libraries and stuff
LIBS=
INCLUDE=
OBJS=
BINS=test_mpi


# Default generic instructions
default:	all
all:	$(OBJS) $(BINS)
clean:	
	rm -f *.o

test_mpi:	test_mpi.c
	$(MPICC) $(CC_FLAGS) -o $@ $<
