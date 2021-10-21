#include "XSbench_header.h"

#ifdef MPI
#include<mpi.h>
#endif

#include <adiak.h>
#include <caliper/cali-manager.h>

int main( int argc, char* argv[] )
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================
	int version = 20;
	int mype = 0;
	double omp_start, omp_end;
	int nprocs = 1;
	unsigned long long verification;

	void *adiak_comm_p = NULL;

	#ifdef MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	adiak_comm_p = &MPI_COMM_NULL;
	#endif

	#ifdef AML
	aml_init(&argc, &argv);
	#endif

	// Process CLI Fields -- store in "Inputs" structure
	Inputs in = read_CLI( argc, argv );

	// Set number of OpenMP Threads
	#ifdef OPENMP
	omp_set_num_threads(in.nthreads); 
	#endif

	adiak_init( adiak_comm_p );

	cali_ConfigManager mgr;
    cali_ConfigManager_new(&mgr);

	if (in.cali_config)
		cali_ConfigManager_add(&mgr, in.cali_config);
    if (cali_ConfigManager_error(&mgr)) {
        cali_SHROUD_array errmsg;
        cali_ConfigManager_error_msg_bufferify(&mgr, &errmsg);
        fprintf(stderr, "Caliper config error: %s\n", errmsg.addr.ccharp);
        cali_SHROUD_memory_destructor(&errmsg.cxx);
    }

    cali_ConfigManager_start(&mgr);

	CALI_MARK_FUNCTION_BEGIN;

	record_globals( in, version );

	// Print-out of Input Summary
	if( mype == 0 )
		print_inputs( in, nprocs, version );

	// =====================================================================
	// Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
	// This is not reflective of a real Monte Carlo simulation workload,
	// therefore, do not profile this region!
	// =====================================================================
	
	SimulationData SD;

	// If read from file mode is selected, skip initialization and load
	// all simulation data structures from file instead
	if( in.binary_mode == READ )
		SD = binary_read(in);
	else
		SD = grid_init_do_not_profile( in, mype );

	// If writing from file mode is selected, write all simulation data
	// structures to file
	if( in.binary_mode == WRITE && mype == 0 )
		binary_write(in, SD);


	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation
	// This is the section that should be profiled, as it reflects a 
	// realistic continuous energy Monte Carlo macroscopic cross section
	// lookup kernel.
	// =====================================================================

	if( mype == 0 )
	{
		printf("\n");
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}

	// Start Simulation Timer
	CALI_MARK_BEGIN("simulation");
	omp_start = get_time();

	// Run simulation
	if( in.simulation_method == EVENT_BASED )
	{
		if( in.kernel_id == 0 )
			verification = run_event_based_simulation(in, SD, mype);
		else if( in.kernel_id == 1 )
			verification = run_event_based_simulation_optimization_1(in, SD, mype);
		else
		{
			printf("Error: No kernel ID %d found!\n", in.kernel_id);
			CALI_MARK_FUNCTION_END;
			exit(1);
		}
	}
	else
		verification = run_history_based_simulation(in, SD, mype);

	if( mype == 0)	
	{	
		printf("\n" );
		printf("Simulation complete.\n" );
	}

	// End Simulation Timer
	omp_end = get_time();
	CALI_MARK_END("simulation");

	// =====================================================================
	// Output Results & Finalize
	// =====================================================================

	// Final Hash Step
	verification = verification % 999983;

	// Print / Save Results and Exit
	int is_invalid_result = print_results( in, mype, omp_end-omp_start, nprocs, verification );

	CALI_MARK_FUNCTION_END;

	cali_ConfigManager_flush(&mgr);
	adiak_fini();

	#ifdef MPI
	MPI_Finalize();
	#endif

	#ifdef AML
	aml_finalize();
	#endif

	return is_invalid_result;
}
