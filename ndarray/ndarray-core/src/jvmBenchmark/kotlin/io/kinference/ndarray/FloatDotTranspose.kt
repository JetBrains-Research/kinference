package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.*
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class FloatDotTranspose {
    @Param("100", "400", "1000")
    var rank: Int = 0
    lateinit var left: FloatNDArray
    lateinit var right: FloatNDArray
    var alpha: Double = 1.0
    lateinit var dest: MutableFloatNDArray
    

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        left = FloatNDArray(FloatTiledArray(strides){ randomFloat()}, strides)
        right = FloatNDArray(FloatTiledArray(strides){ randomFloat()}, strides)
        alpha = Random.nextDouble()
        dest = FloatNDArray.zeros(IntArray(2) { rank })
    }

    @Benchmark
    fun standard(bh: Blackhole) {
        runBlocking {
            left.dotTransposedWithAlpha(alpha, right, dest)
        }
        bh.consume(dest)
    }

    @Benchmark
    fun vectorized(bh: Blackhole) {
        runBlocking {
            left.vectorizedDotTransposedWithAlpha(alpha, right, dest)
        }
        bh.consume(dest)
    }
}
