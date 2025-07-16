package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.softmax.softmax
import kotlin.random.Random
import kotlinx.coroutines.runBlocking
import io.kinference.ndarray.extensions.softmax.softmaxDouble

@State(Scope.Benchmark)
open class DoubleSoftmaxBenchmark {
    @Param("100", "200", "400")
    var rank: Int = 0
    lateinit var src: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray
    lateinit var linearSrc: DoubleLNDArray
    lateinit var linearDest: MutableDoubleLNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(3) { rank })
        src = DoubleNDArray(DoubleTiledArray(strides){ _ -> Random.nextDouble()}, strides)
        dest = DoubleNDArray.zeros(IntArray(3) { rank })
        linearSrc = DoubleLNDArray(strides) { _ : Int -> Random.nextDouble() }
        linearDest = MutableDoubleLNDArray(DoubleArray(strides.linearSize), strides)
    }

    @Benchmark
    fun standardSM(): DoubleNDArray {
        runBlocking {
            softmaxDouble(src, dest, rank, rank*rank)
        }
        return dest
    }

    @Benchmark
    fun linVecSM(): DoubleLNDArray{
        runBlocking {
            vecSoftmax(linearSrc, linearDest, rank, rank*rank)
        }
        return linearDest
    }

    @Benchmark
    fun blkVecSM(): DoubleNDArray{
        runBlocking {
            vecBlkSoftmax(src, dest, rank, rank*rank)
        }
        return dest
    }

}
