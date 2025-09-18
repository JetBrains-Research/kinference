package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.extensions.dot.dotParallelN
import io.kinference.ndarray.extensions.dot.vectorizedDotParallelN
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class FloatDotN {
    @Param("100", "400", "2304")
    var rank: Int = 0

    lateinit var left: FloatNDArray
    lateinit var right: FloatNDArray
    lateinit var dest: MutableFloatNDArray


    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        left = FloatNDArray(FloatTiledArray(strides) { randomFloat() }, strides)
        right = FloatNDArray(FloatTiledArray(strides) { randomFloat() }, strides)
        dest = FloatNDArray.zeros(IntArray(2) { rank })
    }

    @Benchmark
    fun standard(bh: Blackhole) {
        runBlocking {
            dest = dotParallelN(left, right, dest)
        }
        bh.consume(dest)
    }

    @Benchmark
    fun vectorized(bh: Blackhole) {
        runBlocking {
            dest = vectorizedDotParallelN(left, right, dest)
        }
        bh.consume(dest)
    }
}

// max block size: 128
//FloatDotN.standard     100  thrpt    5  19899.920 ± 822.567  ops/s
//FloatDotN.standard     400  thrpt    5    712.032 ±  19.319  ops/s
//FloatDotN.standard    2304  thrpt    5      3.348 ±   0.886  ops/s


// 512
//FloatDotN.standard     100  thrpt    5  19883.242 ± 722.401  ops/s
//FloatDotN.standard     400  thrpt    5    787.289 ± 104.878  ops/s
//FloatDotN.standard    2304  thrpt    5      3.506 ±   0.408  ops/s

// 2048
// FloatDotN.standard    100  thrpt    5  19435.662 ±  632.077  ops/s
// FloatDotN.standard    400  thrpt    5    896.957 ±   25.050  ops/s
// FloatDotN.standard   2304  thrpt    5      5.767 ±    0.574  ops/

// 4096
// FloatDotN.standard   2304  thrpt    5      5.802 ±    0.146  ops/
