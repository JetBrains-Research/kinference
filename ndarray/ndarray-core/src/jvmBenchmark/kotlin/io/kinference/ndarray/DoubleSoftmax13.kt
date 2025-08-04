package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import kotlin.random.Random
import kotlinx.coroutines.runBlocking
import io.kinference.ndarray.extensions.softmax.softmaxVer13Double
import io.kinference.ndarray.extensions.softmax.vecSoftmaxVer13Double

@State(Scope.Benchmark)
open class DoubleSoftmax13 {
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray
    val bh = Blackhole("")

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = DoubleNDArray(DoubleTiledArray(strides) { randomDouble() }, strides)
        dest = DoubleNDArray.zeros(IntArray(3) { rank })
    }

    @Benchmark
    fun standard() {
        runBlocking {
            softmaxVer13Double(src, dest, rank, rank * rank, rank)
        }
        bh.consume(dest)
    }

    @Benchmark
    fun vectorized() {
        runBlocking {
            vecSoftmaxVer13Double(src, dest, rank, rank * rank, rank)
        }
        bh.consume(dest)
    }

}
