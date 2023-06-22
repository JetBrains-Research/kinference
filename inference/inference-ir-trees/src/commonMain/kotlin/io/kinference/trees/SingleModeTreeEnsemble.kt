package io.kinference.trees

import io.kinference.ndarray.arrays.*

abstract class SingleModeTreeEnsemble<T : NumberNDArray>(
    private val aggregator: Aggregator,
    protected val transform: PostTransform,
    private val treeDepths: IntArray,
    private val treeSizes: IntArray,
    featureIds: IntArray,
    nodeFloatSplits: FloatArray,
    private val nonLeafValuesCount: IntArray,
    private val leafValues: FloatArray,
    private val biases: FloatArray,
    val numTargets: Int,
    splitMode: TreeSplitType
) {
    private val treeSplitter = TreeSplitter.get(splitMode, featureIds, nodeFloatSplits)

    protected fun applyEntry(array: FloatArray, output: FloatArray, srcIdx: Int = 0, dstIdx: Int = 0) {
        var index: Int
        var score = FloatArray(numTargets)
        var treeOffset = 0
        var off = 0
        for ((i, depth) in treeDepths.withIndex()) {
            index = 0
            for (j in 1 until depth) {
                index = 2 * index + 1 + treeSplitter.split(array, srcIdx, index + treeOffset)
            }
            off += nonLeafValuesCount[i]
            val treeIndex = (treeOffset + index - off) * numTargets
            score = aggregator.accept(score, leafValues, treeIndex)
            treeOffset += treeSizes[i]
        }
        aggregator.finalize(biases, output, score, dstPosition = dstIdx, numTargets = numTargets)
    }

    abstract suspend fun execute(input: T): T
}
