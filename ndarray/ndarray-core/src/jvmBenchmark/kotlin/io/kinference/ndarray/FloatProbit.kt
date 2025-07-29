package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.probit.probitFloat
import io.kinference.ndarray.extensions.probit.vecProbitFloat
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class FloatProbit {
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: FloatNDArray
    lateinit var dest: FloatNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = FloatNDArray(FloatTiledArray(strides){ _ -> Random.nextFloat()}, strides)
        dest = FloatNDArray.zeros(IntArray(3) { rank })
    }

    @Benchmark
    fun standardSM(): FloatNDArray {
        runBlocking {
            dest = probitFloat(src)
        }
        return dest
    }

    @Benchmark
    fun vectorizedSM(): FloatNDArray {
        runBlocking {
            dest = vecProbitFloat(src)
        }
        return dest
    }

}

// FloatSoftmax13.standard       100  thrpt    5  385.272 ±  15.268  ops/s
// FloatSoftmax13.standard       200  thrpt    5   54.953 ±  68.908  ops/s
// FloatSoftmax13.standard       400  thrpt    5    9.679 ±   0.660  ops/s
// FloatSoftmax13.vectorized     100  thrpt    5  438.411 ±  18.189  ops/s
// FloatSoftmax13.vectorized     200  thrpt    5   63.997 ±  87.344  ops/s
// FloatSoftmax13.vectorized     400  thrpt    5   13.138 ±   0.078  ops/s
// FloatSoftmax13.standard        100  thrpt    5  459.294 ± 482.716  ops/s
// FloatSoftmax13.standard        200  thrpt    5   60.638 ± 112.389  ops/s
// FloatSoftmax13.standard        400  thrpt    5   12.688 ±   0.050  ops/s
// FloatSoftmax13.vectorized      100  thrpt    5  471.050 ± 267.910  ops/s
// FloatSoftmax13.vectorized      200  thrpt    5  100.519 ±  16.921  ops/s
// FloatSoftmax13.vectorized      400  thrpt    5   13.911 ±   1.019  ops/s
