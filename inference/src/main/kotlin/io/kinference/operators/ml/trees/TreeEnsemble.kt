package io.kinference.operators.ml.trees

import io.kinference.ndarray.arrays.*
import io.kinference.protobuf.message.TensorProto

class TreeEnsemble(
    private val aggregator: Aggregator,
    private val transform: PostTransform,
    private val treeDepths: IntArray,
    private val treeSizes: IntArray,
    private val featureIds: IntArray,
    private val nodeFloatSplits: FloatArray,
    private val nonLeafValues: IntArray,
    private val leafValues: FloatArray,
    private val biases: FloatArray,
    private val numTargets: Int,
    internal val labelsInfo: LabelsInfo? = null
) {

    data class LabelsInfo(val labels: List<Any>, val labelsDataType: TensorProto.DataType)

    private fun FloatArray.computeSplit(srcIdx: Int, splitIdx: Int): Int {
        return if (this[srcIdx + featureIds[splitIdx]] > nodeFloatSplits[splitIdx]) 1 else 0
    }

    private fun applyEntry(array: FloatArray, output: MutableFloatNDArray, srcIdx: Int = 0, dstIdx: Int = 0) {
        var index: Int
        var score = FloatArray(numTargets)
        var treeOffset = 0
        var off = 0
        for ((i, depth) in treeDepths.withIndex()) {
            index = 0
            for (j in 1 until depth) {
                index = 2 * index + 1 + array.computeSplit(srcIdx, index + treeOffset)
            }
            off += nonLeafValues[i]
            val treeIndex = (treeOffset + index - off) * numTargets
            score = aggregator.accept(score, leafValues, treeIndex)
            treeOffset += treeSizes[i]
        }
        aggregator.finalize(biases, output.array, dstIdx, score, numTargets)
    }

    fun execute(input: NumberNDArray): NDArray {
        val n = if (input.rank == 1) 1 else input.shape[0]
        val outputShape = if (numTargets == 1) intArrayOf(n) else intArrayOf(n, numTargets)
        val output = MutableFloatNDArray(shape = outputShape)
        input as FloatNDArray

        val leadDim = input.shape[0]
        val array = input.array.toArray()
        if (input.rank == 1 || leadDim == 1) {
            applyEntry(array, output)
        } else {
            val stride = input.shape[1]
            for (i in 0 until leadDim) {
                applyEntry(array, output, srcIdx = i * stride, dstIdx = i * numTargets)
            }
        }
        return transform.apply(output)
    }
}
