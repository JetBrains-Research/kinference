package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.activations.elu.*
import io.kinference.ndarray.extensions.probit.vecProbitDouble
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class DoubleElu {
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: DoubleNDArray
    lateinit var dest: DoubleNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = DoubleNDArray(DoubleTiledArray(strides) { _ -> Random.nextDouble() }, strides)
        dest = DoubleNDArray.zeros(IntArray(3) { rank })
    }

    @Benchmark
    fun standard(): DoubleNDArray {
        runBlocking {
            dest = src.elu()
        }
        return dest
    }

    @Benchmark
    fun vectorized(): DoubleNDArray {
        runBlocking {
            dest = src.vectorizedElu()
        }
        return dest
    }

}

// DoubleSoftmax13.standard       100  thrpt    5  385.272 ±  15.268  ops/s
// DoubleSoftmax13.standard       200  thrpt    5   54.953 ±  68.908  ops/s
// DoubleSoftmax13.standard       400  thrpt    5    9.679 ±   0.660  ops/s
// DoubleSoftmax13.vectorized     100  thrpt    5  438.411 ±  18.189  ops/s
// DoubleSoftmax13.vectorized     200  thrpt    5   63.997 ±  87.344  ops/s
// DoubleSoftmax13.vectorized     400  thrpt    5   13.138 ±   0.078  ops/s
// DoubleSoftmax13.standard        100  thrpt    5  459.294 ± 482.716  ops/s
// DoubleSoftmax13.standard        200  thrpt    5   60.638 ± 112.389  ops/s
// DoubleSoftmax13.standard        400  thrpt    5   12.688 ±   0.050  ops/s
// DoubleSoftmax13.vectorized      100  thrpt    5  471.050 ± 267.910  ops/s
// DoubleSoftmax13.vectorized      200  thrpt    5  100.519 ±  16.921  ops/s
// DoubleSoftmax13.vectorized      400  thrpt    5   13.911 ±   1.019  ops/s
