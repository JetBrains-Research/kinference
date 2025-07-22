package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.softmax.*
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class FloatSoftmax {
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: FloatNDArray
    lateinit var dest: MutableFloatNDArray
    lateinit var linearSrc: FloatLNDArray
    lateinit var linearDest: MutableFloatLNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = FloatNDArray(FloatTiledArray(strides) { _ -> Random.nextFloat() }, strides)
        dest = FloatNDArray.zeros(IntArray(3) { rank })
    }

    @Benchmark
    fun vectorized(): FloatNDArray {
        runBlocking {
            vectorizedSoftmaxFloat(src, dest, rank, rank * rank)
        }
        return dest
    }

    @Benchmark
    fun standard(): FloatNDArray {
        runBlocking {
            softmaxFloat(src, dest, rank, rank * rank)
        }
        return dest
    }

}
