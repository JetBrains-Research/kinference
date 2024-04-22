package io.kinference.trees

import io.kinference.ndarray.arrays.*

abstract class SingleModeTreeEnsemble<T : NumberNDArray>(
    private val aggregator: Aggregator,
    private val treeSizes: IntArray,
    featureIds: IntArray,
    nodeFloatSplits: FloatArray,
    private val nextNodeIds: IntArray,
    private val leafValues: FloatArray,
    private val leafCounter: IntArray,
    private val biases: FloatArray,
    val numTargets: Int,
    splitMode: TreeSplitType
) {
    private val treeSplitter = TreeSplitter.get(splitMode, featureIds, nodeFloatSplits)

    protected fun applyEntry(array: FloatArray, output: FloatArray, srcIdx: Int = 0, dstIdx: Int = 0) {
        var score = FloatArray(numTargets)
        var treeOffset = 0
        for (treeSize in treeSizes) {
            var index = 0
            while (nextNodeIds[2 * (treeOffset + index)] != 0) {
                val split = treeSplitter.split(array, srcIdx, treeOffset + index)
                index = nextNodeIds[2 * (treeOffset + index) + split]
            }

            val leafValueIdx = leafCounter[treeOffset + index] * numTargets
            score = aggregator.accept(score, leafValues, leafValueIdx)

            treeOffset += treeSize
        }
        aggregator.finalize(biases, output, score, dstPosition = dstIdx, numTargets = numTargets)
    }

    abstract suspend fun execute(input: T): T
}
