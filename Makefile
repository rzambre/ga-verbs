CC = gcc
MPICC = mpicc
CFLAGS = -Wall -O3# -g3
MACROS = #-DERRCHK
IFLAGS = -I/home/rzambre/base-rdma-core/build/include
LFLAGS = -L/home/rzambre/base-rdma-core/build/lib
LNAME = -libverbs
OMPFLAGS = -fopenmp
DEPS = shared.c

TARGETS = global_array_effmt global_array_soamt global_array_soamproc

global_array_effmt: global_array_effmt.c $(DEPS)
	$(MPICC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@ $(IFLAGS) $(LFLAGS) $(LNAME)

global_array_soamt: global_array_soamt.c $(DEPS)
	$(MPICC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@ $(IFLAGS) $(LFLAGS) $(LNAME)

global_array_soamproc: global_array_soamproc.c $(DEPS)
	$(MPICC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@ $(IFLAGS) $(LFLAGS) $(LNAME)

clean:
	rm -f $(TARGETS)
