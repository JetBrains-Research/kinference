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

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        left = DoubleNDArray(DoubleTiledArray(strides){ _ -> Random.nextDouble()}, strides)
        right = DoubleNDArray(DoubleTiledArray(strides){ _ -> Random.nextDouble()}, strides)
        alpha = Random.nextDouble()
        dest = DoubleNDArray.zeros(IntArray(2) { rank })
    }

    @Benchmark
    fun standard(): DoubleNDArray {
        runBlocking {
            left.dotTransposedWithAlpha(alpha, right, dest)
        }
        return dest
    }

    @Benchmark
    fun vectorized(): DoubleNDArray {
        runBlocking {
            left.vectorizedDotTransposedWithAlpha(alpha, right, dest)
        }
        return dest
    }
}
