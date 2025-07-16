package io.kinference.ndarray

import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.softmax.softmax
import kotlin.random.Random
import kotlinx.coroutines.runBlocking
import io.kinference.ndarray.extensions.softmax.softmaxFloat

@State(Scope.Benchmark)
open class FloatSoftmaxBenchmark {
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
        linearSrc = FloatLNDArray(strides) { _: Int -> Random.nextFloat() }
        linearDest = MutableFloatLNDArray(FloatArray(strides.linearSize), strides)
    }


    @Benchmark
    fun HelperLinVecSM(): FloatLNDArray {
        runBlocking {
            vecSoftmaxHelper(linearSrc, linearDest, rank, rank * rank)
        }
        return linearDest
    }

//    @Benchmark
//    fun standardSM(): FloatNDArray {
//        runBlocking {
//            softmaxFloat(src, dest, rank, rank * rank)
//        }
//        return dest
//    }
//
//    @Benchmark
//    fun linVecSM(): FloatLNDArray {
//        runBlocking {
//            vecSoftmax(linearSrc, linearDest, rank, rank * rank)
//        }
//        return linearDest
//    }

    //@Benchmark
    //fun classSM(): FloatLNDArray{
    //    runBlocking {
    //        vecSoftmaxClass(linearSrc, linearDest, rank, rank * rank)
    //    }
    //    return linearDest
    //}

    @Benchmark
    fun genSM(): FloatLNDArray{
        runBlocking {
            vecSoftmaxGenerated(linearSrc, linearDest, rank, rank * rank)
        }
        return linearDest
    }
    //@Benchmark
    //fun blkVecSM(): FloatNDArray {
    //    runBlocking {
    //        vecBlkSoftmax(src, dest, rank, rank * rank)
    //    }
    //    return dest
    //}


}
