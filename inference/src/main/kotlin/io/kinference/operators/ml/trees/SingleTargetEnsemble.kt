package io.kinference.operators.ml.trees

import io.kinference.ndarray.arrays.*

class SingleTargetEnsemble(aggregator: Aggregator, transform: PostTransform, treeDepths: IntArray, treeSizes: IntArray, featureIds: IntArray, nodeFloatSplits: FloatArray, leafValues: FloatArray, biases: FloatArray)
    : AbstractTreeEnsemble(aggregator, transform, treeDepths, treeSizes, featureIds, nodeFloatSplits, leafValues, biases, numTargets = 1) {

    private fun applySingle(array: FloatArray, output: MutableFloatNDArray, srcIdx: Int = 0, dstIdx: Int = 0) {
        var index: Int
        var score = 0f
        var treeOffset = 0
        for ((i, depth) in treeDepths.withIndex()) {
            index = 0
            for (j in 1 until depth) {
                index = 2 * index + 1 + array.computeSplit(srcIdx, index + treeOffset)
            }
            score = aggregator.accept(score, leafValues[index + treeOffset])
            treeOffset += treeSizes[i]
        }
        aggregator.finalize(biases[0], output.array, dstIdx, score)
    }

    override fun execute(input: NumberNDArray): NDArray {
        val n = if (input.rank == 1) 1 else input.shape[0]
        val output = MutableFloatNDArray(shape = intArrayOf(n))
        input as FloatNDArray

        val leadDim = input.shape[0]
        val array = input.array.toArray()
        if (input.rank == 1 || leadDim == 1) {
            applySingle(array, output)
        } else {
            val stride = input.shape[1]
            for (i in 0 until leadDim) {
                applySingle(array, output, srcIdx = i * stride, dstIdx = i)
            }
        }
        return transform.apply(output)
    }
}
