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
    @Param("0", "1", "2", "3")
    var type: Int = 0
    val shape = arrayOf(
        intArrayOf(10000, 10000),
        intArrayOf(20000, 20000),
        intArrayOf(10000, 30000),
        intArrayOf(5000, 60000),
    )
    @Param("2048", "4096", "8192", "16384")
    var bs = 0
    lateinit var src: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray


    @Setup
    fun genArrays() = runBlocking {
        MIN_BLOCK_SIZE = bs
        val strides = Strides(shape[type])
        src = DoubleNDArray(DoubleTiledArray(strides) { randomDouble() }, strides)
        dest = DoubleNDArray.zeros(shape[type])
    }

    @Benchmark
    fun standard(bh: Blackhole) {
        runBlocking {
            softmaxDouble(src, dest, shape[type][0], shape[type][1])
        }
        bh.consume(dest)
    }

    @Benchmark
    fun vectorized(bh: Blackhole) {
        runBlocking {
            vectorizedSoftmaxDouble(src, dest, shape[type][0], shape[type][1])
        }
        bh.consume(dest)
    }

}

// DoubleSoftmax13.standard       100  thrpt    5  385.272 ±  15.268  ops/s
// DoubleSoftmax13.standard       200  thrpt    5   54.953 ±  68.908  ops/s
// DoubleSoftmax13.standard       400  thrpt    5    9.679 ±   0.660  ops/s
// DoubleSoftmax13.vectorized     100  thrpt    5  438.411 ±  18.189  ops/s
// DoubleSoftmax13.vectorized     200  thrpt    5   63.997 ±  87.344  ops/s
// DoubleSoftmax13.vectorized     400  thrpt    5   13.138 ±   0.078  ops/s
// FloatSoftmax13.standard        100  thrpt    5  459.294 ± 482.716  ops/s
// FloatSoftmax13.standard        200  thrpt    5   60.638 ± 112.389  ops/s
// FloatSoftmax13.standard        400  thrpt    5   12.688 ±   0.050  ops/s
// FloatSoftmax13.vectorized      100  thrpt    5  471.050 ± 267.910  ops/s
// FloatSoftmax13.vectorized      200  thrpt    5  100.519 ±  16.921  ops/s
// FloatSoftmax13.vectorized      400  thrpt    5   13.911 ±   1.019  ops/s


// DoubleSoftmax.standard         100  thrpt    5  256.336 ± 36.168  ops/s
// DoubleSoftmax.standard         200  thrpt    5   47.665 ±  0.486  ops/s
// DoubleSoftmax.standard         400  thrpt    5    7.475 ±  0.759  ops/s
// DoubleSoftmax.vectorized       100  thrpt    5  343.252 ± 25.629  ops/s
// DoubleSoftmax.vectorized       200  thrpt    5   67.822 ±  0.368  ops/s
// DoubleSoftmax.vectorized       400  thrpt    5   11.269 ±  0.106  ops/s
// DoubleSoftmax13.standard       100  thrpt    5  289.671 ± 10.640  ops/s
// DoubleSoftmax13.standard       200  thrpt    5   48.527 ±  3.768  ops/s
// DoubleSoftmax13.standard       400  thrpt    5    7.712 ±  0.378  ops/s
// DoubleSoftmax13.vectorized     100  thrpt    5  362.637 ± 10.170  ops/s
// DoubleSoftmax13.vectorized     200  thrpt    5   65.068 ±  6.079  ops/s
// DoubleSoftmax13.vectorized     400  thrpt    5   12.376 ±  0.275  ops/s
