package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import kotlin.random.Random
import kotlinx.coroutines.runBlocking

@State(Scope.Benchmark)
open class FloatDotBenchmark {
    @Param("100", "400", "1000")
    var rank: Int = 0
    lateinit var left: FloatNDArray
    lateinit var right: FloatNDArray
    lateinit var dest: MutableFloatNDArray
    lateinit var linearLeft: FloatLNDArray
    lateinit var linearRight: FloatLNDArray
    lateinit var linearDest: MutableFloatLNDArray

    @Setup
    fun genArrays() = runBlocking {
        val strides = Strides(IntArray(2) { rank })
        left = FloatNDArray(FloatTiledArray(strides){ _ -> Random.nextFloat()}, strides)
        right = FloatNDArray(FloatTiledArray(strides){ _ -> Random.nextFloat()}, strides)
        dest = FloatNDArray.zeros(IntArray(2) { rank })
        linearLeft = FloatLNDArray(strides) { _ : Int -> Random.nextFloat() }
        linearRight = FloatLNDArray(strides) { _ : Int -> Random.nextFloat() }
        linearDest = MutableFloatLNDArray(FloatArray(strides.linearSize), strides)
    }

    @Benchmark
    fun standardDot(): FloatNDArray {
        runBlocking {
            left.dot(right as NumberNDArray, dest as MutableNumberNDArray)
        }
        return dest
    }

    @Benchmark
    fun parallelVectorDot(): FloatLNDArray{
        runBlocking {
            dotVectorChunked(linearLeft, linearRight, linearDest)
        }
        return linearDest
    }

    @Benchmark
    fun linearNDArrayDot(): FloatLNDArray {
        runBlocking {
            linearLeft.dot(linearRight, linearDest)
        }

        return linearDest
    }

    @Benchmark
    fun parallelLVDot(): FloatLNDArray {
        runBlocking {
            dotLV(linearLeft, linearRight, linearDest)
        }
        return linearDest
    }

}
