package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.logistic.logisticDouble
import io.kinference.ndarray.extensions.logistic.vecLogisticDouble
import kotlin.random.Random
import kotlinx.coroutines.runBlocking
import io.kinference.ndarray.extensions.softmax.softmaxVer13Double
import io.kinference.ndarray.extensions.softmax.vecSoftmaxVer13Double

@State(Scope.Benchmark)
open class DoubleLogistic {
    @Param("10", "20", "100")
    var rank: Int = 0
    lateinit var src: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray
    

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        src = DoubleNDArray(DoubleTiledArray(strides) { randomDouble() }, strides)
        dest = DoubleNDArray.zeros(IntArray(2) { rank })
    }

    @Benchmark
    fun standard(bh: Blackhole) {
        runBlocking {
            dest = logisticDouble(src,dest)
        }
        bh.consume(dest)
    }

    @Benchmark
    fun vectorized(bh: Blackhole) {
        runBlocking {
            dest = vecLogisticDouble(src,dest)
        }
        bh.consume(dest)
    }

}
