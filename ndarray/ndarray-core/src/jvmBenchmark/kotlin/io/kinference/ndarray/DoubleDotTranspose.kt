package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.*
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class DoubleDotTranspose {
    @Param("100", "400", "1000")
    var rank: Int = 0
    lateinit var left: DoubleNDArray
    lateinit var right: DoubleNDArray
    var alpha: Double = 1.0
    lateinit var dest: MutableDoubleNDArray
    val bh = Blackhole("")

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        left = DoubleNDArray(DoubleTiledArray(strides){ randomDouble()}, strides)
        right = DoubleNDArray(DoubleTiledArray(strides){ randomDouble()}, strides)
        alpha = Random.nextDouble()
        dest = DoubleNDArray.zeros(IntArray(2) { rank })
    }

    @Benchmark
    fun standard() {
        runBlocking {
            left.dotTransposedWithAlpha(alpha, right, dest)
        }
        bh.consume(dest)
    }

    @Benchmark
    fun vectorized() {
        runBlocking {
            left.vectorizedDotTransposedWithAlpha(alpha, right, dest)
        }
        bh.consume(dest)
    }
}
