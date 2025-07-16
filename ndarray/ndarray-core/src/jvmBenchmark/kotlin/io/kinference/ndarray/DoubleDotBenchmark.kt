package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class DoubleDotBenchmark {
    @Param("100", "400", "1000")
    var rank: Int = 0
    lateinit var left: DoubleNDArray
    lateinit var right: DoubleNDArray
    lateinit var dest: MutableDoubleNDArray
    lateinit var linearLeft: DoubleLNDArray
    lateinit var linearRight: DoubleLNDArray
    lateinit var linearDest: MutableDoubleLNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        left = DoubleNDArray(DoubleTiledArray(strides){ _ -> Random.nextDouble()}, strides)
        right = DoubleNDArray(DoubleTiledArray(strides){ _ -> Random.nextDouble()}, strides)
        dest = DoubleNDArray.zeros(IntArray(2) { rank })
        linearLeft = DoubleLNDArray(strides) { _ : Int -> Random.nextDouble() }
        linearRight = DoubleLNDArray(strides) { _ : Int -> Random.nextDouble() }
        linearDest = MutableDoubleLNDArray(DoubleArray(strides.linearSize), strides)
    }

    @Benchmark
    fun standardDot(): DoubleNDArray {
        runBlocking {
            left.dot(right as NumberNDArray, dest as MutableNumberNDArray)
        }
        return dest
    }

    @Benchmark
    fun parallelVectorDot(): DoubleLNDArray{
        runBlocking {
            dotVectorChunked(linearLeft, linearRight, linearDest)
        }
        return linearDest
    }

    @Benchmark
    fun linearNDArrayDot(): DoubleLNDArray {
        runBlocking {
            linearLeft.dot(linearRight, linearDest)
        }

        return linearDest
    }

    @Benchmark
    fun parallelLVDot(): DoubleLNDArray {
        runBlocking {
            dotLV(linearLeft, linearRight, linearDest)
        }
        return linearDest
    }

}

