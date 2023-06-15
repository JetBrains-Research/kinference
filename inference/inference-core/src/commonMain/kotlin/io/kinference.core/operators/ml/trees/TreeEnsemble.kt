package io.kinference.core.operators.ml.trees

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.FloatTiledArray
import io.kinference.primitives.types.DataType
import io.kinference.utils.LoggerFactory
import io.kinference.utils.PlatformUtils
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.*

open class TreeEnsemble(
    private val aggregator: Aggregator,
    private val transform: PostTransform,
    private val treeDepths: IntArray,
    private val treeSizes: IntArray,
    private val featureIds: IntArray,
    private val nodeFloatSplits: FloatArray,
    private val nonLeafValuesCount: IntArray,
    private val leafValues: FloatArray,
    private val biases: FloatArray,
    val numTargets: Int,
) {
    init {
        require(numTargets == 1) { "Only single-target ensembles are supported, but numTargets=$numTargets found" }
    }
    private fun FloatArray.computeSplitGT(splitIdx: Int): Int {
        return if (this[featureIds[splitIdx]] > nodeFloatSplits[splitIdx]) 1 else 0
    }

    private fun applyEntry(array: FloatArray, output: FloatArray) {
        var index: Int
        var score = FloatArray(numTargets)
        var treeOffset = 0
        var off = 0
        for ((i, depth) in treeDepths.withIndex()) {
            index = 0
            for (j in 1 until depth) {
                index = 2 * index + 1 + array.computeSplitGT(index + treeOffset)
            }
            off += nonLeafValuesCount[i]
            val treeIndex = (treeOffset + index - off) * numTargets
            score = aggregator.accept(score, leafValues, treeIndex)
            treeOffset += treeSizes[i]
        }
        aggregator.finalize(biases, output, score, numTargets)
    }

    suspend fun execute(input: NumberNDArray): FloatNDArray {
        val actualInput = reformatInput(input)
        val n = if (input.rank == 1) 1 else input.shape[0]
        val outputShape = if (numTargets == 1) intArrayOf(n) else intArrayOf(n, numTargets)
        val outputBlocks = Array(n) { FloatArray(numTargets) }

        val leadDim = actualInput.shape[0]
        val arrayBlocks = actualInput.array.blocks

        if (input.rank == 1 || leadDim == 1) {
            applyEntry(arrayBlocks[0], outputBlocks[0])
        } else {
            val batchSize = ceil(leadDim.toFloat() / PlatformUtils.threads).toInt()

            coroutineScope {
                for (i in 0 until leadDim step batchSize) {
                    launch {
                        for (j in i until min(i + batchSize, leadDim))
                            applyEntry(arrayBlocks[j], outputBlocks[j])
                    }
                }
            }
        }
        val output = MutableFloatNDArray(FloatTiledArray(outputBlocks), Strides(outputShape))
        return transform.apply(output)
    }

    companion object {
        private val logger = LoggerFactory.create("io.kinference.core.operators.ml.trees.TreeEnsemble")

        private fun DoubleNDArray.toFloatNDArray(): FloatNDArray {
            val pointer = this.array.pointer()
            return FloatNDArray.matrixLike(this.shape) { pointer.getAndIncrement().toFloat() }
        }

        private fun FloatNDArray.resizeBlock(): FloatNDArray {
            val inputPointer = this.array.pointer()
            return FloatNDArray.matrixLike(this.shape) { inputPointer.getAndIncrement() }
        }

        fun reformatInput(input: NumberNDArray): FloatNDArray {
            require(input.type == DataType.DOUBLE || input.type == DataType.FLOAT) { "Integer inputs are not supported yet" }

            if (input is FloatNDArray && input.array.blockSize == input.shape.last()) return input

            return if (input is DoubleNDArray) {
                logger.warning { "Using inputs of type ${input.type} may cause performance to slow down. Use inputs of FLOAT type instead" }
                input.toFloatNDArray()
            } else {
                input as FloatNDArray
                // Input is 2D Array. For optimal performance, inner tiled array should be represented
                // as an array of matrix rows to process the matrix row by row.
                logger.warning { "Inner block size is not equal to input row size. Use FloatNDArray.matrixLike(...) array constructor to adjust block size." }
                input.resizeBlock()
            }
        }
    }
}
