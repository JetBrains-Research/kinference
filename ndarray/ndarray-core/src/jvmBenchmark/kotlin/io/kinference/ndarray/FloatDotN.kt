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
    @Param("100", "400", "1000")
    var rank: Int = 0
    lateinit var left: FloatNDArray
    lateinit var right: FloatNDArray
    lateinit var dest: MutableFloatNDArray
    

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        left = FloatNDArray(FloatTiledArray(strides){ randomFloat()}, strides)
        right = FloatNDArray(FloatTiledArray(strides){ randomFloat()}, strides)
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
