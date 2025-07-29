package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.logistic.logisticFloat
import io.kinference.ndarray.extensions.logistic.vecLogisticFloat
import kotlin.random.Random
import kotlinx.coroutines.runBlocking
import io.kinference.ndarray.extensions.softmax.softmaxVer13Float
import io.kinference.ndarray.extensions.softmax.vecSoftmaxVer13Float

@State(Scope.Benchmark)
open class FloatLogistic {
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: FloatNDArray
    lateinit var dest: MutableFloatNDArray
    lateinit var linearSrc: FloatLNDArray
    lateinit var linearDest: MutableFloatLNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = FloatNDArray(FloatTiledArray(strides) { _ -> Random.nextFloat() }, strides)
        dest = FloatNDArray.zeros(IntArray(3) { rank })
    }

    @Benchmark
    fun standard(): FloatNDArray {
        runBlocking {
            dest = logisticFloat(src,dest)
        }
        return dest
    }

    @Benchmark
    fun vectorized(): FloatNDArray {
        runBlocking {
            dest = vecLogisticFloat(src,dest)
        }
        return dest
    }

}
