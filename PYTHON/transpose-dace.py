#!/usr/bin/env python3

import sys
print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer

import numpy
print('Numpy version  =', numpy.version.version)
import dace
print('DaCe version =', dace.__version__)
print()

def main():

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Python Numpy Matrix transpose: B = A^T')

    if len(sys.argv) < 4:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./transpose <# iterations> <matrix order> <tile size> [collapse]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    order = int(sys.argv[2])
    if order < 1:
        sys.exit("ERROR: order must be >= 1")

    tile_size = int(sys.argv[3])
    if tile_size < 1:
        sys.exit("ERROR: tile size must be >= 1")

    collapse = False
    if len(sys.argv) >= 5:
        collapse = bool(sys.argv[4])

    print('Number of iterations =', iterations)
    print('Matrix order         =', order)
    print('Tile size            =', tile_size)
    print('Loop collapse        =', collapse)
    print()

    print('creating and compiling function...')

    # defining program
    @dace.program
    def transpose(A: dace.float64[order, order],
                  B: dace.float64[order, order],
                  iterations: dace.int32):
        for k in range(0,iterations):
            for i, j in dace.map[0:order, 0:order]:
                with dace.tasklet:
                    a_in << A[i, j]
                    b_in << B[j, i]
                    a_out >> A[i, j]
                    b_out >> B[j, i]

                    b_out = a_in + b_in
                    a_out = a_in + 1.0

    # convert to SDFG
    sdfg = transpose.to_sdfg()

    # if neccessary apply map tiling
    # (usually done in visual editor)
    if tile_size > 1:
        from dace.transformation.dataflow import MapTiling
        sdfg.apply_transformations(MapTiling, options={'tile_sizes': (tile_size, tile_size)})

    # set collaps property of maps, during code generation this will add
    # collaps(2) to the the '#pragma omp parallel for'
    # (usually done in visual editor)
    if collapse:
        from dace.sdfg.nodes import MapEntry
        for state in sdfg.nodes():
            for node in state:
                if type(node) == MapEntry:
                    node.map.collapse = 2

    # adding instumentation, to measue how long transposition takes
    for state in sdfg.nodes():
        state.instrument = dace.InstrumentationType.Timer
    
    # generate and compile code
    compiled_sdfg = sdfg.compile()

    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    print("initializing input data...")

    A = numpy.fromfunction(lambda i,j: i*order+j, (order,order), dtype=numpy.float64)
    B = numpy.zeros((order,order), dtype=numpy.float64)

    # ********************************************************************
    # ** Transposing matrix
    # ********************************************************************

    print("begin...")
    compiled_sdfg(A=A, B=B, iterations=iterations+1)
    print("end...\n")

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    # result validation
    A = numpy.fromfunction(lambda i,j: ((iterations/2.0)+(order*j+i))*(iterations+1.0), (order,order), dtype=float)
    abserr = numpy.linalg.norm(numpy.reshape(B-A,order*order),ord=1)
    epsilon=1.e-8
    if abserr < epsilon:
        print('Solution validates')
    else:
        print('error ',abserr, ' exceeds threshold ',epsilon)
        sys.exit("ERROR: solution did not validate")

    # performance results
    nbytes = 2 * order**2 * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
    if sdfg.is_instrumented():
        report = sdfg.get_latest_report()
        avgtime = (sum(list(report.durations.values())[0][1:])/iterations) / 1000
        print('Rate (MB/s): ', 1.e-6 * nbytes / avgtime, ' Avg time (s): ', avgtime)


if __name__ == '__main__':
    main()
