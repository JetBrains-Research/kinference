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
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray
    lateinit var linearSrc: DoubleLNDArray
    lateinit var linearDest: MutableDoubleLNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = DoubleNDArray(DoubleTiledArray(strides) { _ -> Random.nextDouble() }, strides)
        dest = DoubleNDArray.zeros(IntArray(3) { rank })
    }

    @Benchmark
    fun standard(): DoubleNDArray {
        runBlocking {
            dest = logisticDouble(src,dest)
        }
        return dest
    }

    @Benchmark
    fun vectorized(): DoubleNDArray {
        runBlocking {
            dest = vecLogisticDouble(src,dest)
        }
        return dest
    }

}
