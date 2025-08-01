package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import kotlin.random.Random
import kotlinx.coroutines.runBlocking
import io.kinference.ndarray.extensions.softmax.softmaxVer13Float
import io.kinference.ndarray.extensions.softmax.vecSoftmaxVer13Float

@State(Scope.Benchmark)
open class FloatSoftmax13 {
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: FloatNDArray
    lateinit var dest: MutableFloatNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = FloatNDArray(FloatTiledArray(strides) { randomFloat() }, strides)
        dest = FloatNDArray.zeros(IntArray(3) { rank })
    }

    @Benchmark
    fun standard(): FloatNDArray {
        runBlocking {
            softmaxVer13Float(src, dest, rank, rank * rank, rank)
        }
        return dest
    }

    @Benchmark
    fun vectorized(): FloatNDArray {
        runBlocking {
            vecSoftmaxVer13Float(src, dest, rank, rank * rank, rank)
        }
        return dest
    }

}
