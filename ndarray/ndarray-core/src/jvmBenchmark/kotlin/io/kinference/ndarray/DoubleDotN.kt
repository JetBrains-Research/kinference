package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.extensions.dot.DotUtils
import io.kinference.ndarray.extensions.dot.dotParallelN
import io.kinference.ndarray.extensions.dot.lcDot
import io.kinference.ndarray.extensions.dot.vectorizedDotParallelN
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class DoubleDotN {
    @Param("0", "1", "2", "3", "4", "5", "6", "7")
    var type = 0
    lateinit var left: DoubleNDArray
    lateinit var right: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray
    lateinit var shape: IntArray
    val shapes = arrayOf(
        intArrayOf(100, 100),
        intArrayOf(400, 400),
        intArrayOf(1000, 1000),
        intArrayOf(1000, 1000),
        intArrayOf(256, 768),
        intArrayOf(768, 256),
        intArrayOf(2304, 768),
        intArrayOf(768, 2304),
    )


    @Setup
    fun genArrays() = runBlocking {
        shape = shapes[type]
        val strides = Strides(shape)
        val rStrides = Strides(shape.reversedArray())
        left = DoubleNDArray(DoubleTiledArray(strides) { randomDouble() }, strides)
        right = DoubleNDArray(DoubleTiledArray(rStrides) { randomDouble() }, rStrides)
        dest = DoubleNDArray.zeros(IntArray(2) { shape[0] })
    }

    @Benchmark
    fun standard(bh: Blackhole) {
        runBlocking {
            dest = dotParallelN(left, right, dest)
        }
        bh.consume(dest)
    }

    @Benchmark
    fun chunk(bh: Blackhole) {
        runBlocking {
            dest = lcDot(left, right, dest)
        }
        bh.consume(dest)
    }
}
