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
open class DoubleDotN {
    @Param("100", "400", "1000")
    var rank: Int = 0
    lateinit var left: DoubleNDArray
    lateinit var right: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        left = DoubleNDArray(DoubleTiledArray(strides){ randomDouble()}, strides)
        right = DoubleNDArray(DoubleTiledArray(strides){ randomDouble()}, strides)
        dest = DoubleNDArray.zeros(IntArray(2) { rank })
    }

    @Benchmark
    fun standard(): DoubleNDArray {
        runBlocking {
            dest = dotParallelN(left, right, dest)
        }
        return dest
    }

    @Benchmark
    fun vectorized(): DoubleNDArray {
        runBlocking {
            dest = vectorizedDotParallelN(left, right, dest)
        }
        return dest
    }
}
