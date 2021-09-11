#!/usr/bin/env python3
#
# Copyright (c) 2015, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#*******************************************************************
#
# NAME:    dgemm
#
# PURPOSE: This program tests the efficiency with which a dense matrix
#          dense multiplication is carried out
#
# USAGE:   The program takes as input the matrix order,
#          the number of times the matrix-matrix multiplication
#          is carried out.
#
#          <progname> <# iterations> <matrix order>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: Written by Rob Van der Wijngaart, February 2009.
#          Converted to Python by Jeff Hammond, February 2016.
#          Fixed timing err, Ave+std_dev, more pythonic, Tim Mattson May 2021
# *******************************************************************

import dace
import numpy as np
import sys

print('Python version = ',
      str(sys.version_info.major) + '.' + str(sys.version_info.minor))
if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer


def main():

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels version ')  #, PRKVERSION
    print('Python Dense matrix-matrix multiplication: C = A x B')

    if len(sys.argv) != 3:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./dgemm <# iterations> <matrix order>")

    iters = int(sys.argv[1])
    if iters < 1:
        sys.exit("ERROR: iterations must be >= 1")

    order = int(sys.argv[2])
    if order < 1:
        sys.exit("ERROR: order must be >= 1")

    print('Number of iterations = ', iters)
    print('Matrix order         = ', order)

    # ********************************************************************
    # setup and compile program
    # ********************************************************************

    # define DaCe program
    @dace.program
    def dgemm(A: dace.float64[order, order], B: dace.float64[order, order],
              C: dace.float64[order, order], iterations: dace.int32):
        for iter in range(0, iterations):
            for i, j, k in dace.map[0:order, 0:order, 0:order]:
                C[i, j] = A[i, k] * B[k, j] + C[i, j]

    # convert program to SDFG (dataflow graph)
    sdfg = dgemm.to_sdfg()

    # tile map, prevent false sharing of cache lines
    from dace.transformation.dataflow import MapTiling
    block_size = 16
    sdfg.apply_transformations(
        MapTiling,
        options={'tile_sizes': (block_size, block_size, block_size)})

    # sdfg.optimize()

    # activate instrumentation feature
    for state in sdfg.nodes():
        state.instrument = dace.InstrumentationType.Timer

    # compile SDFG to program executable
    compiled_sdfg = sdfg.compile()

    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    A = np.fromfunction(lambda i, j: j, (order, order), dtype=float)
    B = np.fromfunction(lambda i, j: j, (order, order), dtype=float)
    C = np.zeros((order, order))

    # Call compiled DaCe program
    compiled_sdfg(A=A, B=B, C=C, iterations=np.int32(iters + 1))

    # Extract measurements

    # Use DaCe internal instrumentation features
    if sdfg.is_instrumented():
        report = sdfg.get_latest_report()
        timings = np.array(list(report.durations.values())[0][1:]) / 1000
        dgemmAve = np.mean(timings)
        dgemmStdDev = np.std(timings)
    else:
        print("Instrumentation failed")
        dgemmAve = 0.0
        dgemmStdDev = 1.0

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    checksum = 0.0
    for i in range(order):
        for j in range(order):
            checksum += C[i][j]

    ref_checksum = 0.25 * order * order * order * (order - 1.0) * (order - 1.0)
    ref_checksum *= (iters + 1)

    epsilon = 1.e-8
    if abs((checksum - ref_checksum) / ref_checksum) < epsilon:
        print('Solution validates')
        nflops = 2.0 * order * order * order
        recipDiff = (1.0 / (dgemmAve - dgemmStdDev) - 1.0 /
                     (dgemmAve + dgemmStdDev))
        GfStdDev = 1.e-6 * nflops * recipDiff / 2.0
        print('nflops: ', nflops)
        print('Rate: ', 1.e-6 * nflops / dgemmAve, ' +/- (MF/s): ', GfStdDev)
    else:
        print('ERROR: Checksum = ', checksum, ', Reference checksum = ',
              ref_checksum, '\n')
        sys.exit("ERROR: solution did not validate")


if __name__ == '__main__':
    main()
