package io.kinference

import io.kinference.ndarray.Strides
import kotlin.random.Random
import kotlin.test.*
import kotlin.time.*

class MatMulTest {

    private fun dotBaseline(left: FloatArray, right: FloatArray, dest: FloatArray, m: Int, n: Int, t: Int) {
        for (i in 0 until n) {
            val dInd = i * m
            val lInd = i * t
            for (k in 0 until t) {
                val temp = left[lInd + k]
                val rInd = k * m
                for (j in 0 until m) {
                    dest[dInd + j] += temp * right[rInd + j]
                }
            }
        }
    }

    private fun dotTiledBaseline(left: TiledArray, right: TiledArray, dest: TiledArray, m: Int, n: Int, t: Int) {
        val rdBlockSize = dest.blockSize
        for (rdCol in 0 until right.blocksInRow) {
            val rightIdx = rdCol * t
            val destIdx = rdCol * n

            for (i in 0 until n) {
                val destBlock = dest.blocks[destIdx + i]

                for (lCol in 0 until left.blocksInRow) {
                    val leftBlock = left.blocks[i + lCol * n]
                    val rightIdxOffset = rightIdx + left.blockSize * lCol

                    for (k in 0 until left.blockSize) {
                        val temp = leftBlock[k]
                        val rightBlock = right.blocks[rightIdxOffset + k]

                        for (j in 0 until rdBlockSize) {
                            destBlock[j] += temp * rightBlock[j]
                        }
                    }
                }
            }
        }
    }

    @OptIn(ExperimentalTime::class)
    @Test
    fun test() {
        val random = Random(42)

        val n = 512
        val m = 512
        val t = 512

        val count = 10

        println("Start matrix multiplication test M: $m, N: $n, T: $t:")

        val left = FloatArray(n * t) { random.nextFloat() }
        val right = FloatArray(t * m) { random.nextFloat() }
        var destBaseline = FloatArray(n * m)

        val baselineTime = measureTime {
            repeat(count) {
                destBaseline = FloatArray(n * m)
                dotBaseline(left, right, destBaseline, m, n, t)
            }
        }

        println("Baseline: ${baselineTime / count}")

        var counter = 0
        val leftTiled = TiledArray(Strides(intArrayOf(n, t))) { left[counter++] }

        counter = 0
        val rightTiled = TiledArray(Strides(intArrayOf(t, m))) { right[counter++] }

        var destTiledBaseline = TiledArray(Strides(intArrayOf(n, m))) { 0f }

        val tiledBaselineTime = measureTime {
            repeat(count) {
                destTiledBaseline = TiledArray(Strides(intArrayOf(n, m))) { 0f }
                dotTiledBaseline(leftTiled, rightTiled, destTiledBaseline, m, n, t)
            }
        }

        println("Tiled Baseline: ${tiledBaselineTime / count}")

        assertTrue(destBaseline.contentEquals(destTiledBaseline.toArray()))
    }
}
