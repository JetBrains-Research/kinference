package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.gelu.computeGeluDouble
import io.kinference.ndarray.extensions.gelu.vecGeluDouble
import io.kinference.utils.inlines.InlineInt
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class DoubleBiasGelu {
    @Param("2304", "768", "1000")
    var rank: Int = 0
    lateinit var src: DoubleNDArray
    lateinit var bias: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        src = DoubleNDArray(DoubleTiledArray(strides) { randomDouble() }, strides)
        bias = DoubleNDArray(DoubleTiledArray(strides) { randomDouble() }, strides)
        dest = MutableDoubleNDArray(strides)
    }

    @Benchmark
    fun standard(bh: Blackhole) {
        runBlocking {
            dest = computeGeluDouble(src, bias, dest)
        }
        bh.consume(dest)
    }

    @Benchmark
    fun vectorized(bh: Blackhole) {
        runBlocking {
            dest = vecGeluDouble(src, bias, dest)
        }
        bh.consume(dest)
    }
}
