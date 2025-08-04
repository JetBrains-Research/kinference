package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.gelu.fastGeluFloat
import io.kinference.ndarray.extensions.gelu.vecFastGeluFloat
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class FloatFastGelu {
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: FloatNDArray
    lateinit var dest: MutableFloatNDArray
    val bh = Blackhole("")

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = FloatNDArray(FloatTiledArray(strides) { randomFloat() }, strides)
    }

    @Benchmark
    fun standard() {
        runBlocking {
            dest = fastGeluFloat(src, null)
        }
        bh.consume(dest)
    }

    @Benchmark
    fun vectorized() {
        runBlocking {
            dest = vecFastGeluFloat(src, null)
        }
        bh.consume(dest)
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
