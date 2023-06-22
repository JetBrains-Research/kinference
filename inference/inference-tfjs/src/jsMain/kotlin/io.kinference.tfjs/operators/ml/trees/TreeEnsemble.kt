package io.kinference.tfjs.operators.ml.trees

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.dataFloat
import io.kinference.primitives.types.DataType
import io.kinference.utils.LoggerFactory

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
    val numTargets: Int
) {
    private fun FloatArray.computeSplitGT(srcIdx: Int, splitIdx: Int): Int {
        return if (this[srcIdx + featureIds[splitIdx]] > nodeFloatSplits[splitIdx]) 1 else 0
    }

    private fun applyEntry(array: FloatArray, output: FloatArray, srcIdx: Int = 0, dstIdx: Int = 0) {
        var index: Int
        var score = FloatArray(numTargets)
        var treeOffset = 0
        var off = 0
        for ((i, depth) in treeDepths.withIndex()) {
            index = 0
            for (j in 1 until depth) {
                index = 2 * index + 1 + array.computeSplitGT(srcIdx, index + treeOffset)
            }
            off += nonLeafValuesCount[i]
            val treeIndex = (treeOffset + index - off) * numTargets
            score = aggregator.accept(score, leafValues, treeIndex)
            treeOffset += treeSizes[i]
        }
        aggregator.finalize(biases, output, dstIdx, score, numTargets)
    }

    suspend fun execute(input: NumberNDArrayTFJS): NumberNDArrayTFJS {
        require(input.type == DataType.DOUBLE || input.type == DataType.FLOAT) { "Integer inputs are not supported yet" }

        val inputArray = input.tfjsArray.dataFloat()
        val n = if (input.rank == 1) 1 else input.shape[0]
        val outputShape = if (numTargets == 1) arrayOf(n) else arrayOf(n, numTargets)
        val outputArray = FloatArray(n * numTargets)

        val leadDim = input.shape[0]

        if (input.rank == 1 || leadDim == 1) {
            applyEntry(inputArray, outputArray)
        } else {
            val stride = input.shape[1]
            for (i in 0 until leadDim) {
                applyEntry(inputArray, outputArray, srcIdx = i * stride, dstIdx = i * numTargets)
            }
        }
        val output = NDArrayTFJS.float(outputArray, outputShape).asMutable()
        return transform.apply(output)
    }

    companion object {
        private val logger = LoggerFactory.create("io.kinference.tfjs.operators.ml.trees.TreeEnsemble")
    }
}
