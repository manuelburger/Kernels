#!/usr/bin/env python3
#
# Copyright (c) 2017, Intel Corporation
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
# NAME:    nstream
#
# PURPOSE: To compute memory bandwidth when adding a vector of a given
#          number of double precision values to the scalar multiple of
#          another vector of the same length, and storing the result in
#          a third vector.
#
# USAGE:   The program takes as input the number
#          of iterations to loop over the triad vectors, the length of the
#          vectors, and the offset between vectors
#
#          <progname> <# iterations> <vector length> <offset>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# NOTES:   Bandwidth is determined as the number of words read, plus the
#          number of words written, times the size of the words, divided
#          by the execution time. For a vector length of N, the total
#          number of words read and written is 4*N*sizeof(double).
#
#
# HISTORY: This code is loosely based on the Stream benchmark by John
#          McCalpin, but does not follow all the Stream rules. Hence,
#          reported results should not be associated with Stream in
#          external publications
#
#          Converted to Python by Jeff Hammond, October 2017.
#
# *******************************************************************

import sys

print('Python version = ',
      str(sys.version_info.major) + '.' + str(sys.version_info.minor))
if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer

import dace
import numpy

print('Numpy version  = ', numpy.version.version)


def main():

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels version ')  #, PRKVERSION
    print('Python STREAM triad: A = B + scalar * C')

    if len(sys.argv) != 3:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: python nstream.py <# iterations> <vector length>")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    length = int(sys.argv[2])
    if length < 1:
        sys.exit("ERROR: length must be positive")

    #offset = int(sys.argv[3])
    #if offset < 0:
    #    sys.exit("ERROR: offset must be nonnegative")

    print('Number of iterations = ', iterations)
    print('Vector length        = ', length)
    #print('Offset               = ', offset)

    # ********************************************************************
    # setup and compile program
    # ********************************************************************

    # define DaCe program
    @dace.program
    def nstream(A: dace.float64[length], B: dace.float64[length],
                C: dace.float64[length], scalar: dace.float64,
                iterations: dace.int32):
        for k in range(0, iterations):
            A[:] = A + B + scalar * C

    # convert program to SDFG (dataflow graph)
    sdfg = nstream.to_sdfg()

    # fuse maps
    from dace.transformation.dataflow import MapFusion, MapTiling
    sdfg.apply_transformations_repeated([MapFusion])

    # tile map, prevent false sharing of cache lines
    sdfg.apply_transformations(MapTiling, options={'tile_sizes': (8192, )})

    # activate instrumentation feature
    for state in sdfg.nodes():
        state.instrument = dace.InstrumentationType.Timer

    # compile SDFG to program executable
    compiled_sdfg = sdfg.compile()

    # ********************************************************************
    # ** Allocate space for the input and execute STREAM triad
    # ********************************************************************

    # 0.0 is a float, which is 64b (53b of precision)
    A = numpy.zeros(length, dtype=numpy.float64)
    B = numpy.full(length, 2.0, dtype=numpy.float64)
    C = numpy.full(length, 2.0, dtype=numpy.float64)

    scalar = numpy.float64(3.0)

    # Call DaCe compiled program
    compiled_sdfg(A=A,
                  B=B,
                  C=C,
                  scalar=scalar,
                  iterations=numpy.int32(iterations + 1))

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    ar = 0.0
    br = 2.0
    cr = 2.0
    ref = 0.0
    for k in range(0, iterations + 1):
        ar += br + scalar * cr

    ar *= length

    asum = 0.0
    for i in range(length):
        asum += abs(A[i])

    epsilon = 1.e-8
    if abs(ar - asum) / asum > epsilon:
        print('Failed Validation on output array')
        print('        Expected checksum: ', ar)
        print('        Observed checksum: ', asum)
        sys.exit("ERROR: solution did not validate")
    else:
        print('Solution validates')

        # Use DaCe internal instrumentation features
        if sdfg.is_instrumented():

            report = sdfg.get_latest_report()
            avgtime = (sum(list(report.durations.values())[0][1:]) /
                       iterations) / 1000

            nbytes = 4.0 * length * 8  # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
            print('Rate (MB/s): ', 1.e-6 * nbytes / avgtime, ' Avg time (s): ',
                  avgtime)
        else:
            print("Instrumentation failed")


if __name__ == '__main__':
    main()
