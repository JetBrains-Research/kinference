package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import kotlin.random.Random
import kotlinx.coroutines.runBlocking
import io.kinference.ndarray.extensions.softmax.softmaxDouble
import io.kinference.ndarray.extensions.softmax.vectorizedSoftmaxDouble

@State(Scope.Benchmark)
open class DoubleSoftmax {
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray
    lateinit var linearSrc: DoubleLNDArray
    lateinit var linearDest: MutableDoubleLNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = DoubleNDArray(DoubleTiledArray(strides){ _ -> Random.nextDouble()}, strides)
        dest = DoubleNDArray.zeros(IntArray(3) { rank })
    }

    @Benchmark
    fun standardSM(): DoubleNDArray {
        runBlocking {
            softmaxDouble(src, dest, rank, rank*rank)
        }
        return dest
    }

    @Benchmark
    fun vectorizedSM(): DoubleNDArray {
        runBlocking {
            vectorizedSoftmaxDouble(src, dest, rank, rank*rank)
        }
        return dest
    }

}
