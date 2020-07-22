#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <time.h>

#include <mpi.h>
#include <omp.h>
#include <infiniband/verbs.h>

#include "shared.h"

/**
 * Rank 1 will contain the global arrays A, B, and C.
 * Rank 0 will RDMA-read tiles from the global array and RDMA-write back
 * the tiles into the global array.
 * The global array is a square matrix.
 * The tile can be rectangular.
 * The network configuration is similar to the one in
 * state-of-the-art hybrid MPI i.e. all threads wil use only 1 endpoint
 * for communication.
 */

int init_params(void);

int read_args(int argc, char *argv[]);

static int get_tile(int tile_y, int tile_x, unsigned long int global_addr, int global_rkey,
					int v_tile_dim, int hz_tile_dim, int hz_ga_dim, struct tile *loc_tile,
					int element_bytes, struct thread_flow_vars *flow_vars, struct thread_flow_vars *flow_vars_arr);

static int put_tile(int tile_y, int tile_x, unsigned long int global_addr, int global_rkey,
					int v_tile_dim, int hz_tile_dim, int hz_ga_dim, struct tile *loc_tile,
					int element_bytes, struct thread_flow_vars *flow_vars, struct thread_flow_vars *flow_vars_arr);

static int wait(struct thread_flow_vars *flow_vars, struct thread_flow_vars *flow_vars_arr);

static int init_ep_res(void);

static int connect_eps(void);

static int free_ep_res(void);

struct ibv_context *dev_context;
struct ibv_pd *pd;
struct ibv_mr *global_a_mr;
struct ibv_mr *global_b_mr;
struct ibv_mr *global_c_mr;
struct tile *tile_a;
struct tile *tile_b;
struct tile *tile_c;
struct ibv_cq *cq;
struct ibv_qp *qp;

int rem_tx_depth;

int main (int argc, char *argv[])
{
	int ret = 0, provided;

	double *global_a; // rank 1
	double *global_b; // rank 1
	double *global_c; // rank 1

	int global_a_rkey;
	int global_b_rkey;
	int global_c_rkey;

	unsigned long int global_a_addr;
	unsigned long int global_b_addr;
	unsigned long int global_c_addr;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE) {
		printf("Insufficient threading support\n");
		ret = EXIT_FAILURE;
		goto clean;
	}
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size > 2) {
		fprintf(stderr, "Supporting only two processes at the moment\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	ret = init_params();
	if (ret) {
		fprintf(stderr, "Error in initializing paramaters\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}

	ret = read_args(argc, argv);
	if (ret)
		goto clean_mpi;

	if ((ga_dim_x % tile_dim_x != 0) || (ga_dim_y % tile_dim_y != 0) || (ga_dim_common % tile_dim_common != 0)) {
		fprintf(stderr, "Tile dimensions must be a factor of the corresponding dimensions of the global array\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}
	
	if ((tile_dim_y % postlist != 0) || (tile_dim_common % postlist != 0)) {
		fprintf(stderr, "Postlist must be a factor of both tile_dim_y and tile_dim_common \n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}

	omp_set_num_threads(num_threads);

	ret = init_ep_res();
	if (ret) {
		fprintf(stderr, "Failure in initializing EP's resources.\n");
		goto clean_mpi;
	}

	ret = connect_eps();
	if (ret) {
		fprintf(stderr, "Failure in connecting eps.\n");
		goto clean_mpi;
	}

	if (rank) {
		/* Allocate memory for the global arrays */
		int global_a_size = ga_dim_y * ga_dim_common;
		int global_b_size = ga_dim_common * ga_dim_x;
		int global_c_size = ga_dim_y * ga_dim_x;

		ret = posix_memalign((void**) &global_a, CACHE_LINE_SIZE, global_a_size * sizeof *global_a);
		if (ret) {
			fprintf(stderr, "Error in posix_memalign of global A array.\n");
			goto clean_mpi;
		}
		memset(global_a, 0, global_a_size * sizeof *global_a);

		ret = posix_memalign((void**) &global_b, CACHE_LINE_SIZE, global_b_size * sizeof *global_b);
		if (ret) {
			fprintf(stderr, "error in posix_memalign of global b array.\n");
			goto clean_mpi;
		}
		memset(global_b, 0, global_b_size * sizeof *global_b);
		
		ret = posix_memalign((void**) &global_c, CACHE_LINE_SIZE, global_c_size * sizeof *global_c);
		if (ret) {
			fprintf(stderr, "error in posix_memalign of global b array.\n");
			goto clean_mpi;
		}
		memset(global_c, 0, global_c_size * sizeof *global_c);

		/* Register the global arrays for NIC access */
		global_a_mr = ibv_reg_mr(pd, global_a, global_a_size * sizeof *global_a, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
		if (!global_a_mr) {
			fprintf(stderr, "Error in registering global A array\n");
			ret = EXIT_FAILURE;
			goto clean_mpi;
		}
		global_a_addr = (unsigned long int) global_a_mr->addr;

		global_b_mr = ibv_reg_mr(pd, global_b, global_b_size * sizeof *global_b, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
		if (!global_b_mr) {
			fprintf(stderr, "Error in registering global B array\n");
			ret = EXIT_FAILURE;
			goto clean_mpi;
		}
		global_b_addr = (unsigned long int) global_b_mr->addr;
		
		global_c_mr = ibv_reg_mr(pd, global_c, global_c_size * sizeof *global_c, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
		if (!global_c_mr) {
			fprintf(stderr, "Error in registering global C array\n");
			ret = EXIT_FAILURE;
			goto clean_mpi;
		}
		global_c_addr = (unsigned long int) global_c_mr->addr;

		/* Send rank 0 the rkeys and addresses of the global arrays */
		MPI_Send(&global_a_mr->rkey, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&global_b_mr->rkey, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&global_c_mr->rkey, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&global_a_addr, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&global_b_addr, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&global_c_addr, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
	} else {
		/* Allocate memory for each of the tiles */
		ret = posix_memalign((void**) &tile_a, CACHE_LINE_SIZE, num_threads * sizeof *tile_a);
		if (ret) {
			fprintf(stderr, "Error in posix_memalign of struct tiles for A.\n");
			goto clean_mpi;
		}

		ret = posix_memalign((void**) &tile_b, CACHE_LINE_SIZE, num_threads * sizeof *tile_b);
		if (ret) {
			fprintf(stderr, "Error in posix_memalign of struct tiles for A.\n");
			goto clean_mpi;
		}

		ret = posix_memalign((void**) &tile_c, CACHE_LINE_SIZE, num_threads * sizeof *tile_c);
		if (ret) {
			fprintf(stderr, "Error in posix_memalign of struct tiles for A.\n");
			goto clean_mpi;
		}

		int tile_a_size = tile_dim_y * tile_dim_common;
		int tile_b_size = tile_dim_common * tile_dim_x;
		int tile_c_size = tile_dim_y * tile_dim_x;
		
		int tid;
		for (tid = 0; tid < num_threads; tid++) {
			/* Allocate the array for each tile */
			ret = posix_memalign((void**) &tile_a[tid].tile_arr, CACHE_LINE_SIZE, tile_a_size * sizeof *tile_a[tid].tile_arr);
			if (ret) {
				fprintf(stderr, "Error in posix_memalign of tile array A.\n");
				goto clean_mpi;
			}
			memset(tile_a[tid].tile_arr, 0, tile_a_size * sizeof *tile_a[tid].tile_arr);

			ret = posix_memalign((void**) &tile_b[tid].tile_arr, CACHE_LINE_SIZE, tile_b_size * sizeof *tile_b[tid].tile_arr);
			if (ret) {
				fprintf(stderr, "Error in posix_memalign of tile array A.\n");
				goto clean_mpi;
			}
			memset(tile_b[tid].tile_arr, 0, tile_b_size * sizeof *tile_b[tid].tile_arr);

			ret = posix_memalign((void**) &tile_c[tid].tile_arr, CACHE_LINE_SIZE, tile_c_size * sizeof *tile_c[tid].tile_arr);
			if (ret) {
				fprintf(stderr, "Error in posix_memalign of tile array A.\n");
				goto clean_mpi;
			}
			memset(tile_c[tid].tile_arr, 0, tile_c_size * sizeof *tile_c[tid].tile_arr);
			
			/* Register the arrays of tiles for NIC access */
			tile_a[tid].mr = ibv_reg_mr(pd, tile_a[tid].tile_arr, tile_a_size * sizeof *tile_a[tid].tile_arr, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
			if (!tile_a[tid].mr) {
				fprintf(stderr, "Error in registering tile A\n");
				ret = EXIT_FAILURE;
				goto clean_mpi;
			}

			tile_b[tid].mr = ibv_reg_mr(pd, tile_b[tid].tile_arr, tile_b_size * sizeof *tile_b[tid].tile_arr, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
			if (!tile_b[tid].mr) {
				fprintf(stderr, "Error in registering tile B\n");
				ret = EXIT_FAILURE;
				goto clean_mpi;
			}

			tile_c[tid].mr = ibv_reg_mr(pd, tile_c[tid].tile_arr, tile_c_size * sizeof *tile_c[tid].tile_arr, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
			if (!tile_c[tid].mr) {
				fprintf(stderr, "Error in registering tile C\n");
				ret = EXIT_FAILURE;
				goto clean_mpi;
			}
		}

		/* Receive the rkeys and the addresses of the global arrays from rank 1 */
		MPI_Recv(&global_a_rkey, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&global_b_rkey, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&global_c_rkey, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&global_a_addr, 1, MPI_UNSIGNED_LONG, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&global_b_addr, 1, MPI_UNSIGNED_LONG, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&global_c_addr, 1, MPI_UNSIGNED_LONG, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	if (!rank) {

		int *th_read_messages, *th_write_messages;
		int *th_read_a_counter, *th_read_b_counter, *th_write_c_counter;
		int tile_a_bytes, tile_b_bytes, tile_c_bytes;
		int element_bytes;
		double *th_read_time, *th_write_time;

		th_read_messages = calloc(num_threads, sizeof *th_read_messages);
		th_write_messages = calloc(num_threads, sizeof *th_write_messages);
		th_read_a_counter = calloc(num_threads, sizeof *th_read_a_counter);
		th_read_b_counter = calloc(num_threads, sizeof *th_read_b_counter);
		th_write_c_counter = calloc(num_threads, sizeof *th_write_c_counter);
		th_read_time = calloc(num_threads, sizeof *th_read_time);
		th_write_time = calloc(num_threads, sizeof *th_write_time);

		element_bytes = sizeof(double); // TODO: keep it configurable

		struct thread_flow_vars *flow_vars = malloc(num_threads * sizeof(struct thread_flow_vars));
		if (!flow_vars) {
			fprintf(stderr, "Failure in allocating array of thread_flow_vars\n");
			goto clean_mpi;
		}

		rem_tx_depth = tx_depth;
		int decrement_val = max(postlist, mod_comp);
		
		#pragma omp parallel private(ret) firstprivate(element_bytes) 
		{
			int tid, tile_id;
			int p;
			int num_tiles, num_tiles_y, num_tiles_common, num_tiles_x;
			int read_messages, write_messages;
			int read_a_counter, read_b_counter, write_c_counter;

			double read_start, write_start;
			double read_time, write_time;

			struct ibv_sge *SGE;
			struct ibv_send_wr *send_wqe;
			struct ibv_wc *WC;
			
			posix_memalign((void**) &SGE, CACHE_LINE_SIZE, postlist * sizeof(struct ibv_sge));
			posix_memalign((void**) &send_wqe, CACHE_LINE_SIZE, postlist * sizeof(struct ibv_send_wr));
			posix_memalign( (void**) &WC, CACHE_LINE_SIZE, cq_depth * sizeof(struct ibv_wc) );
			
			memset(SGE, 0, postlist * sizeof(struct ibv_sge));
			memset(send_wqe, 0, postlist * sizeof(struct ibv_send_wr));
			memset(WC, 0, postlist * sizeof(struct ibv_wc));
			
			tid = omp_get_thread_num();

			struct ibv_qp_attr attr;
			struct ibv_qp_init_attr qp_init_attr;
			ret = ibv_query_qp(qp, &attr, IBV_QP_CAP, &qp_init_attr);
			if (ret) {
				fprintf(stderr, "Failure in querying the QP\n");
				exit(0);
			}		
			
			/* prepare values that will not change */
			for (p = 0; p < postlist; p++) {
				send_wqe[p].wr_id = tid;
				send_wqe[p].next = (p == postlist-1) ? NULL: &send_wqe[p+1];
				send_wqe[p].sg_list = &SGE[p];
				send_wqe[p].num_sge = 1;
			}

			struct thread_flow_vars *all_flow_vars = flow_vars;
			struct thread_flow_vars *my_flow_vars = &flow_vars[tid];
			my_flow_vars->tid = tid;
			my_flow_vars->post_count = 0;
			my_flow_vars->comp_count = 0;
			my_flow_vars->posts = 0;
			my_flow_vars->postlist = postlist;
			my_flow_vars->mod_comp = mod_comp;
			my_flow_vars->tx_depth = tx_depth;
			my_flow_vars->cq_depth = cq_depth;
			my_flow_vars->tx_decrement_val = decrement_val;
			my_flow_vars->max_inline_data = qp_init_attr.cap.max_inline_data;
			my_flow_vars->sge = SGE;
			my_flow_vars->wqe = send_wqe;
			my_flow_vars->wc = WC;
			my_flow_vars->my_qp = qp;
			my_flow_vars->my_cq = cq;
			
			num_tiles_y = ga_dim_y / tile_dim_y;
			num_tiles_common = ga_dim_common / tile_dim_common;
			num_tiles_x = ga_dim_x / tile_dim_x;
			num_tiles = num_tiles_common * num_tiles_x * num_tiles_y;
			
			read_time = 0;
			write_time = 0;
			read_messages = 0;
			write_messages = 0;
			read_a_counter = 0;
			read_b_counter = 0;
			write_c_counter = 0;
			
			for (tile_id = tid; tile_id < num_tiles; tile_id+=num_threads) {
				int tile_i = tile_id / (num_tiles_common * num_tiles_x);
				int tile_j = (tile_id / num_tiles_common) % num_tiles_x;
				int tile_k = tile_id % num_tiles_common;

				/* Get tile from global array A */
				read_start = MPI_Wtime();
				ret = get_tile(tile_i, tile_k, global_a_addr, global_a_rkey,
							tile_dim_y, tile_dim_common, ga_dim_common, &tile_a[tid],
							element_bytes, my_flow_vars, all_flow_vars);
				#ifdef ERRCHK
				if (ret) {
					fprintf(stderr, "Error in getting tile from A\n");
					exit(0);
				}
				#endif
				/* Get tile from global array B */
				ret = get_tile(tile_k, tile_j, global_b_addr, global_b_rkey,
							tile_dim_common, tile_dim_x, ga_dim_x, &tile_b[tid],
							element_bytes, my_flow_vars, all_flow_vars);
				#ifdef ERRCHK
				if (ret) {
					fprintf(stderr, "Error in getting tile from B\n");
					exit(0);
				}
				#endif
				/* Complete the Get */
				ret = wait(my_flow_vars, all_flow_vars);
				#ifdef ERRCHK
				if (ret) {
					fprintf(stderr, "Error in completing pending operations\n");
					exit(0);
				}
				#endif
				read_time += MPI_Wtime() - read_start;
				read_messages += tile_dim_y + tile_dim_common;
				read_a_counter++;
				read_b_counter++;
				
				/* Compute */
				if (compute)
					dgemm(tile_a[tid].tile_arr, tile_b[tid].tile_arr, tile_c[tid].tile_arr,
							tile_dim_y, tile_dim_common, tile_dim_x);

				/* Put tile into global array C */
				write_start = MPI_Wtime();
				ret = put_tile(tile_i, tile_j, global_c_addr, global_c_rkey,
							tile_dim_y, tile_dim_x, ga_dim_x, &tile_c[tid],
							element_bytes, my_flow_vars, all_flow_vars);
				#ifdef ERRCHK
				if (ret) {
					fprintf(stderr, "Error in putting tile into C\n");
					exit(0);	
				}
				#endif

				/* Complete the Put */
				ret = wait(my_flow_vars, all_flow_vars);
				#ifdef ERRCHK
				if (ret) {
					fprintf(stderr, "Error in completing pending operations\n");
					exit(0);
				}
				#endif
				write_time += MPI_Wtime() - write_start;
				write_messages += tile_dim_y;
				write_c_counter++;
			}

			th_read_messages[tid] = read_messages;
			th_write_messages[tid] = write_messages;
			th_read_a_counter[tid] = read_a_counter;
			th_read_b_counter[tid] = read_b_counter;
			th_write_c_counter[tid] = write_c_counter;
			th_read_time[tid] = read_time;
			th_write_time[tid] = write_time;
			free(SGE);
			free(send_wqe);
			free(WC);
		}

		tile_a_bytes = tile_dim_y * tile_dim_common * element_bytes;
		tile_b_bytes = tile_dim_common * tile_dim_x * element_bytes;
		tile_c_bytes = tile_dim_y * tile_dim_x * element_bytes;

		show_perf(th_read_messages, th_write_messages,
				th_read_a_counter, th_read_b_counter, th_write_c_counter,
				tile_a_bytes, tile_b_bytes, tile_c_bytes,
				th_read_time, th_write_time,
				num_threads);
		
		free(th_read_messages);
		free(th_write_messages);
		free(th_read_a_counter);
		free(th_read_b_counter);
		free(th_write_c_counter);
		free(th_read_time);
		free(th_write_time);

		MPI_Barrier(MPI_COMM_WORLD);
	} else {
		MPI_Barrier(MPI_COMM_WORLD);
	}

	ret = free_ep_res();
	if (ret)
		fprintf(stderr, "Failure in freeing resources\n");
	
	if (rank) {
		free(global_a);
		free(global_b);
		free(global_c);
	} else {
		int tid;
		for (tid = 0; tid < num_threads; tid++) {
			free(tile_a[tid].tile_arr);
			free(tile_b[tid].tile_arr);
			free(tile_c[tid].tile_arr);
		}
		free(tile_a);
		free(tile_b);
		free(tile_c);
	}

clean_mpi:
	MPI_Finalize();
clean:
	return ret;
}

static int get_tile(int tile_y, int tile_x, unsigned long int global_addr, int global_rkey,
					int v_tile_dim, int hz_tile_dim, int hz_ga_dim, struct tile *loc_tile,
					int element_bytes, struct thread_flow_vars *flow_vars, struct thread_flow_vars *flow_vars_arr)
{
	int ret = 0;
	int p;
	int comp_tid, cqe_count;
	int cur_rem_tx_depth;
	int bytes_in_wqe, bytes_in_ga_row;
	uint64_t local_base_addr, remote_base_addr;
	struct ibv_send_wr *bad_send_wqe;
	int cqe_i;

	flow_vars->posts += v_tile_dim;
	
	bytes_in_wqe = hz_tile_dim * element_bytes;
	bytes_in_ga_row = hz_ga_dim * element_bytes;

	for (p = 0; p < flow_vars->postlist; p++) {
		flow_vars->sge[p].length	= bytes_in_wqe;
		flow_vars->sge[p].lkey		= loc_tile->mr->lkey;

		flow_vars->wqe[p].opcode = IBV_WR_RDMA_READ;
		// TODO: will send_inline help at all in RDMA_read?
		flow_vars->wqe[p].wr.rdma.rkey = global_rkey;
	}

	local_base_addr = (uint64_t) loc_tile->mr->addr;
	remote_base_addr = (uint64_t) global_addr + (tile_y * hz_ga_dim + tile_x * hz_tile_dim) * element_bytes;

	while (flow_vars->post_count < flow_vars->posts) {
		#pragma omp atomic capture
		{cur_rem_tx_depth = rem_tx_depth; rem_tx_depth -= flow_vars->tx_decrement_val;}
		if (cur_rem_tx_depth <= 0) {
			#pragma omp atomic
			rem_tx_depth += flow_vars->tx_decrement_val;
			goto poll;
		}
		do { // the number of doorbells is dependendent on the remaining QP-depth
			for (p = 0; p < flow_vars->postlist; p++) {
				flow_vars->sge[p].addr 	= local_base_addr;
				
				if ((flow_vars->post_count+p+1) % flow_vars->mod_comp == 0)
					flow_vars->wqe[p].send_flags = IBV_SEND_SIGNALED;
				else
					flow_vars->wqe[p].send_flags = 0;
				flow_vars->wqe[p].wr.rdma.remote_addr = remote_base_addr;
				
				local_base_addr += bytes_in_wqe;
				remote_base_addr += bytes_in_ga_row; 
			}
			ret = ibv_post_send(flow_vars->my_qp, &flow_vars->wqe[0], &bad_send_wqe);
			#ifdef ERRCHK
			if (ret) {
				fprintf(stderr, "Thread %d: Error %d in posting send_wqe on QP\n", flow_vars->tid, ret);
				goto exit;
			}
			#endif
			flow_vars->post_count += flow_vars->postlist;
		} while (flow_vars->post_count % flow_vars->mod_comp); // in the case where mod_comp > postlist and the thread hasn't posted mod_comp posts yet
poll:
		if (cur_rem_tx_depth <= 0) { // poll only if SQ is full
			cqe_count = ibv_poll_cq(flow_vars->my_cq, flow_vars->cq_depth, flow_vars->wc);
			#ifdef ERRCHK
			if (cqe_count < 0) {
				fprintf(stderr, "Thread %d: Failure in polling CQ: %d\n", flow_vars->tid, cqe_count);
				ret = cqe_count;
				goto exit;
			}
			#endif
			for (cqe_i = 0; cqe_i < cqe_count; cqe_i++) {
				#ifdef ERRCHK
				if (flow_vars->wc[cqe_i].status != IBV_WC_SUCCESS) {
					fprintf(stderr, "Thread %d: Failed status %s for %d; cqe_count %d\n", flow_vars->tid,
							ibv_wc_status_str(flow_vars->wc[cqe_i].status), (int) flow_vars->wc[cqe_i].wr_id, cqe_i);
					ret = EXIT_FAILURE;
					goto exit;
				}
				#endif
				comp_tid = flow_vars->wc[cqe_i].wr_id;
				#pragma omp atomic
				flow_vars_arr[comp_tid].comp_count += flow_vars_arr[comp_tid].mod_comp;
				#pragma omp atomic
				rem_tx_depth += flow_vars_arr[comp_tid].mod_comp;	
			}
		}
	}

#ifdef ERRCHK
exit:
#endif
	return ret;
}

static int wait(struct thread_flow_vars *flow_vars, struct thread_flow_vars *flow_vars_arr)
{
	int ret = 0;
	int comp_tid, cqe_count;
	int cqe_i;

	while (flow_vars->comp_count < flow_vars->post_count) {
		cqe_count = ibv_poll_cq(flow_vars->my_cq, flow_vars->cq_depth, flow_vars->wc);
		#ifdef ERRCHK
		if (cqe_count < 0) {
			fprintf(stderr, "Thread %d: Failure in polling CQ: %d\n", flow_vars->tid, cqe_count);
			ret = cqe_count;
			goto exit; 
		}
		#endif
		for (cqe_i = 0; cqe_i < cqe_count; cqe_i++) {
			#ifdef ERRCHK
			if (flow_vars->wc[cqe_i].status != IBV_WC_SUCCESS) {
				fprintf(stderr, "Thread %d: Failed status %s for %d; cqe_count %d\n", flow_vars->tid,
						ibv_wc_status_str(flow_vars->wc[cqe_i].status), (int) flow_vars->wc[cqe_i].wr_id, cqe_i);
				ret = EXIT_FAILURE;
				goto exit;
			}
			#endif
			comp_tid = flow_vars->wc[cqe_i].wr_id;
			#pragma omp atomic
			flow_vars_arr[comp_tid].comp_count += flow_vars_arr[comp_tid].mod_comp;
			#pragma omp atomic
			rem_tx_depth += flow_vars_arr[comp_tid].mod_comp;
		}
	}

#ifdef ERRCHK
exit:
#endif
	return ret;
}

static int put_tile(int tile_y, int tile_x, unsigned long int global_addr, int global_rkey,
					int v_tile_dim, int hz_tile_dim, int hz_ga_dim, struct tile *loc_tile,
					int element_bytes, struct thread_flow_vars *flow_vars, struct thread_flow_vars *flow_vars_arr)
{
	int ret = 0;
	int p, db, doorbells;
	int cqe_count;
	int send_inline;
	int bytes_in_wqe, bytes_in_ga_row;
	uint64_t local_base_addr, remote_base_addr;
	struct ibv_send_wr *bad_send_wqe;
	#ifdef ERRCHK
	int cqe_i;
	#endif
	
	flow_vars->posts += v_tile_dim;
	
	send_inline = ((hz_tile_dim * element_bytes) <= flow_vars->max_inline_data) ? IBV_SEND_INLINE : 0;
	bytes_in_wqe = hz_tile_dim * element_bytes;
	bytes_in_ga_row = hz_ga_dim * element_bytes;

	for (p = 0; p < flow_vars->postlist; p++) {
		flow_vars->sge[p].length	= bytes_in_wqe;
		flow_vars->sge[p].lkey		= loc_tile->mr->lkey;

		flow_vars->wqe[p].opcode = IBV_WR_RDMA_WRITE;
		flow_vars->wqe[p].send_flags = send_inline;
		flow_vars->wqe[p].wr.rdma.rkey = global_rkey;
	}

	local_base_addr = (uint64_t) loc_tile->mr->addr;
	remote_base_addr = (uint64_t) global_addr + (tile_y * hz_ga_dim + tile_x * hz_tile_dim) * element_bytes;
				
	while (flow_vars->post_count < flow_vars->posts) {
		// Post
		doorbells = min((flow_vars->posts - flow_vars->post_count),
						(flow_vars->tx_depth - (flow_vars->post_count - flow_vars->comp_count))) / flow_vars->postlist;
		for (db = 0; db < doorbells; db++) {
			for (p = 0; p < flow_vars->postlist; p++) {
				flow_vars->sge[p].addr 	= local_base_addr;
				
				if ((flow_vars->post_count+p+1) % flow_vars->mod_comp == 0)
					flow_vars->wqe[p].send_flags = IBV_SEND_SIGNALED;
				else
					flow_vars->wqe[p].send_flags = 0;
				flow_vars->wqe[p].send_flags |= send_inline;
				flow_vars->wqe[p].wr.rdma.remote_addr = remote_base_addr;
				
				local_base_addr += bytes_in_wqe;
				remote_base_addr += bytes_in_ga_row; 
			}
			ret = ibv_post_send(flow_vars->my_qp, &flow_vars->wqe[0], &bad_send_wqe);
			#ifdef ERRCHK
			if (ret) {
				fprintf(stderr, "Thread %d: Error %d in posting send_wqe on QP\n", flow_vars->tid, ret);
				goto exit;
			}
			#endif
			flow_vars->post_count += flow_vars->postlist;
		}
		if (!doorbells) {
			// Poll only if SQ is full
			cqe_count = ibv_poll_cq(flow_vars->my_cq, flow_vars->cq_depth, flow_vars->wc);
			#ifdef ERRCHK
			if (cqe_count < 0) {
				fprintf(stderr, "Thread %d: Failure in polling CQ: %d\n", flow_vars->tid, cqe_count);
				ret = cqe_count;
				goto exit;
			}
			for (cqe_i = 0; cqe_i < cqe_count; cqe_i++) {
				if (flow_vars->wc[cqe_i].status != IBV_WC_SUCCESS) {
					fprintf(stderr, "Thread %d: Failed status %s for %d; cqe_count %d\n", flow_vars->tid,
							ibv_wc_status_str(flow_vars->wc[cqe_i].status), (int) flow_vars->wc[cqe_i].wr_id, cqe_i);
					ret = EXIT_FAILURE;
					goto exit;
				}
			}
			#endif
			flow_vars->comp_count += (cqe_count * flow_vars->mod_comp);
		}
	}

#ifdef ERRCHK
exit:
#endif
	return ret;
}

static int init_ep_res(void)
{
	int ret = 0;
	int dev_i;

	struct ibv_device **dev_list;
	struct ibv_device *dev;

	dev_list = ibv_get_device_list(NULL);
	if (!dev_list) {
		fprintf(stderr, "Failed to get IB devices list");
		ret = EXIT_FAILURE;
		goto exit;
	}

	if (!dev_name) {
		dev = *dev_list;
		if (!dev) {
			fprintf(stderr, "No IB devices found\n");
			return EXIT_FAILURE;
		}
	} else {
		for (dev_i = 0; dev_list[dev_i]; ++dev_i)
			if (!strcmp(ibv_get_device_name(dev_list[dev_i]), dev_name))
				break;
		dev = dev_list[dev_i];
		if (!dev) {
			fprintf(stderr, "IB device %s not found\n", dev_name);
			return EXIT_FAILURE;
		}
	}

	/* Acquire a Device Context */	
	dev_context = ibv_open_device(dev);
	if (!dev_context) {
		fprintf(stderr, "Couldn't get context for %s\n",
		ibv_get_device_name(dev));
		ret = EXIT_FAILURE;
		goto clean_dev_list;
	}
	
	/* Open up a Protection Domain */
	pd = ibv_alloc_pd(dev_context);
	if (!pd) {
		fprintf(stderr, "Couldn't allocate PD\n");
		ret = EXIT_FAILURE;
		goto clean_dev_context;
	}

	/* Create Completion Queues */
	cq_depth = tx_depth / mod_comp; // RDMA operations require checks only on SQ's WQEs
	cq = ibv_create_cq(dev_context, cq_depth, NULL, NULL, 0);
	if (!cq) {
		fprintf(stderr, "Couldn't create CQ\n");
		ret = EXIT_FAILURE;
		goto clean_pd;
	}

	/* Create Queue Pairs and transition them to the INIT state */
	struct ibv_qp_init_attr qp_init_attr = {
		.send_cq = cq,
		.recv_cq = cq, // the same CQ for sending and receiving
		.cap     = {
			.max_send_wr  = tx_depth, // maximum number of outstanding WRs that can be posted to the SQ in this QP
			.max_recv_wr  = rx_depth, // maximum number of outstanding WRs that can be posted to the RQ in this QP
			.max_send_sge = 1,
			.max_recv_sge = 1,
		},
		.qp_type = IBV_QPT_RC,
		.sq_sig_all = 0, // all send_wqes posted will generate a WC
	};
	
	qp = ibv_create_qp(pd, &qp_init_attr); // this puts the QP in the RESET state
	if (!qp) {
		fprintf(stderr, "Couldn't create QP\n");
		ret = EXIT_FAILURE;
		goto clean_cq;
	}

	struct ibv_qp_attr qp_attr = {
		.qp_state = IBV_QPS_INIT,
		.pkey_index = 0, // according to examples
		.port_num = ib_port,
		.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
	};

	/* Initialize the QP to the INIT state */
	ret = ibv_modify_qp(qp, &qp_attr,
				IBV_QP_STATE		|
				IBV_QP_PKEY_INDEX	|
				IBV_QP_PORT 		|
				IBV_QP_ACCESS_FLAGS);
	if (ret) {
		fprintf(stderr, "Failed to modify QP to INIT\n");
		goto clean_qp;
	}

	goto exit;

clean_qp:
	ibv_destroy_qp(qp);
	
clean_cq:
	ibv_destroy_cq(cq);

clean_pd:
	ibv_dealloc_pd(pd);

clean_dev_context:
	ibv_close_device(dev_context);

clean_dev_list:
	ibv_free_device_list(dev_list);

exit:
	return ret;
}

static int connect_eps(void)
{
	int ret = 0;
	int target, my_lid, dest_lid;
	int dest_qp_index;

	/* Query port to get my LID */
	struct ibv_port_attr ib_port_attr;
	ret = ibv_query_port(dev_context, ib_port, &ib_port_attr);
	if (ret) {
		fprintf(stderr, "Failed to get port info\n");
		ret = EXIT_FAILURE;
		goto exit;
	}
	my_lid = ib_port_attr.lid;

	if (!rank) {
		target = 1;
		
		MPI_Send(&my_lid, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
		MPI_Recv(&dest_lid, 1, MPI_INT, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Send(&qp->qp_num, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
		MPI_Recv(&dest_qp_index, 1, MPI_INT, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	} else {
		target = 0;

		MPI_Recv(&dest_lid, 1, MPI_INT, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&my_lid, 1, MPI_INT, target, 0, MPI_COMM_WORLD);

		MPI_Recv(&dest_qp_index, 1, MPI_INT, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&qp->qp_num, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
	}

	/* Transition to RTR state */
	struct ibv_qp_attr qp_attr = {
		.qp_state = IBV_QPS_RTR,
		.ah_attr = {
			.is_global = 0,
			.dlid = dest_lid,
			.sl = 0, // set Service Level to 0 (relates to QoS)
			.src_path_bits = 0,
			.port_num = ib_port,
		},
		.path_mtu = IBV_MTU_4096,
		.dest_qp_num = dest_qp_index,
		.rq_psn = 0,
		.max_dest_rd_atomic = 16, // according to Anuj's benchmark: 16
		.min_rnr_timer = 12
	};

	ret = ibv_modify_qp(qp, &qp_attr,
				IBV_QP_STATE 			|
				IBV_QP_AV			|
				IBV_QP_PATH_MTU			|
				IBV_QP_DEST_QPN			|
				IBV_QP_RQ_PSN			|
				IBV_QP_MAX_DEST_RD_ATOMIC	|
				IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		fprintf(stderr, "Error in transitioning QP to RTR state\n");
		ret = EXIT_FAILURE;
		goto exit;
	}

	/* Transition to RTS state */
	memset(&qp_attr, 0, sizeof(qp_attr)); // reset
	qp_attr.qp_state = IBV_QPS_RTS;
	qp_attr.sq_psn = 0;
	qp_attr.timeout = 14; 
	qp_attr.retry_cnt = 7;
	qp_attr.rnr_retry = 7;
	qp_attr.max_rd_atomic = 16; // according to Anuj's benchmark: 16

	ret = ibv_modify_qp(qp, &qp_attr,
				IBV_QP_STATE		|
				IBV_QP_SQ_PSN		|
				IBV_QP_TIMEOUT		|
				IBV_QP_RETRY_CNT 	|
				IBV_QP_RNR_RETRY	|
				IBV_QP_MAX_QP_RD_ATOMIC );
	if (ret) {
		fprintf(stderr, "Error in transitioning QP to RTS state\n");
		ret = EXIT_FAILURE;
		goto exit;
	}

exit: 
	return ret;
}

int free_ep_res(void)
{
	int tid;

	ibv_destroy_qp(qp);
	ibv_destroy_cq(cq);
	if (rank) {
		ibv_dereg_mr(global_a_mr);
		ibv_dereg_mr(global_b_mr);
		ibv_dereg_mr(global_c_mr);
	} else {
		for (tid = 0; tid < num_threads; tid++) {
			ibv_dereg_mr(tile_a[tid].mr);
			ibv_dereg_mr(tile_b[tid].mr);
			ibv_dereg_mr(tile_c[tid].mr);
		}
	}
	ibv_dealloc_pd(pd);
	ibv_close_device(dev_context);

	return 0;
}

int read_args(int argc, char *argv[])
{
	int ret = 0, op;
	int bench_type;

	struct option long_options[] = {
		{.name = "ib-dev",			.has_arg = 1, .val = 'd'},
		{.name = "num-threads",		.has_arg = 1, .val = 't'},
		{.name = "ga-dim-x",		.has_arg = 1, .val = 'n'},
		{.name = "ga-dim-common",	.has_arg = 1, .val = 'c'},
		{.name = "ga-dim-y",		.has_arg = 1, .val = 'm'},
		{.name = "tile-dim-x",		.has_arg = 1, .val = 'j'},
		{.name = "tile-dim-common",	.has_arg = 1, .val = 'k'},
		{.name = "tile-dim-y",		.has_arg = 1, .val = 'i'},
		{.name = "postlist",		.has_arg = 1, .val = 'p'},
		{.name = "mod-comp",		.has_arg = 1, .val = 'q'},
		{.name = "tx-depth",		.has_arg = 1, .val = 'T'},
		{.name = "rx-depth",		.has_arg = 1, .val = 'R'},
		{.name = "use-cq-ex",		.has_arg = 0, .val = 'x'},
		{.name = "compute",			.has_arg = 0, .val = 'C'},
		//{.name = "use-td",		.has_arg = 0, .val = 'o'},
		{0 , 0 , 0, 0}
	};

	while (1) {
		op = getopt_long(argc, argv, "h?d:t:n:c:m:j:k:i:p:q:T:R:xC", long_options, NULL);
		if (op == -1)
			break;
		
		switch (op) {
			case '?':
			case 'h':
				print_usage(argv[0], 1);
				ret = -1;
				goto exit;
			default:
				parse_args(op, optarg);
				break;
		}
	}

	bench_type = SOA_MT;
	if (optind < argc) {
		print_usage(argv[0], bench_type);
		ret = -1;
	}

exit:
	return ret;
}

int init_params(void)
{
	dev_name = NULL;
	num_threads = DEF_NUM_THREADS;
	
	ga_dim_x = DEF_GA_DIM_X;
	ga_dim_common = DEF_GA_DIM_COMMON;
	ga_dim_y = DEF_GA_DIM_Y;
	tile_dim_x = DEF_TILE_DIM_X;
	tile_dim_common = DEF_TILE_DIM_COMMON;
	tile_dim_y = DEF_TILE_DIM_Y;
	
	postlist = DEF_POSTLIST;
	mod_comp = DEF_MOD_COMP;
	tx_depth = DEF_TX_DEPTH;
	rx_depth = DEF_RX_DEPTH;
	use_cq_ex = DEF_USE_CQ_EX;

	compute = DEF_COMPUTE;

	ib_port = DEF_IB_PORT;

	return 0;
}
