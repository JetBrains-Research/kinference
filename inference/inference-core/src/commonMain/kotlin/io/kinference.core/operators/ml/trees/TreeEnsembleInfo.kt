package io.kinference.core.operators.ml.trees

import io.kinference.ndarray.toIntArray
import kotlin.math.ceil
import kotlin.math.log2

class TreeEnsembleInfo(
    val baseValues: FloatArray?,
    val featureIds: LongArray,
    val nodeModes: List<String>,
    val nodeIds: LongArray,
    val treeIds: LongArray,
    val falseNodeIds: LongArray,
    val trueNodeIds: LongArray,
    val nodeValues: FloatArray,
    postTransform: String?,
    aggregator: String?,
    val targetIds: LongArray,
    val targetNodeIds: LongArray,
    val targetNodeTreeIds: LongArray,
    val targetWeights: FloatArray,
    val numTargets: Int = 1
) {
    private val postTransform = PostTransformType.valueOf(postTransform ?: DEFAULT_POST_TRANSFORM)
    private val aggregator = AggregatorType.valueOf(aggregator ?: DEFAULT_AGGREGATOR)

    fun buildEnsemble(): TreeEnsemble {
        require(numTargets > 0) { "Number of targets should be > 0, got $numTargets" }

        val distinctModes = nodeModes.toHashSet()
        require(distinctModes == SUPPORTED_MODES) { "Got unsupported tree ensemble node modes: ${distinctModes - SUPPORTED_MODES}" }

        val numTrees = treeIds.distinct().size
        val treeDepths = IntArray(numTrees)
        val treeSizes = IntArray(numTrees)
        val nonLeafValuesCount = IntArray(numTrees)

        //count the number of nodes and compute the depth of each tree
        val trees2nodes = treeIds.zip(nodeIds).groupBy { it.first }
        for (entry in trees2nodes) {
            val treeId = entry.key.toInt()
            val currentNumNodes = entry.value.size
            treeSizes[treeId] = currentNumNodes
            treeDepths[treeId] = treeDepthFromNodesNum(currentNumNodes)
        }

        //count non-leaf nodes of each tree
        for (entry in trees2nodes) {
            val treeId = entry.key.toInt()
            //skip first previous trees
            val treeOffset = treeSizes.take(treeId).fold(0, Int::plus)
            for (tree2node in entry.value) {
                val nodeOffset = tree2node.second.toInt() + treeOffset
                if (nodeModes[nodeOffset] != "LEAF") nonLeafValuesCount[treeId]++
            }
        }

        return TreeEnsemble(
            aggregator = Aggregator[aggregator],
            transform = PostTransform[postTransform],
            treeDepths = treeDepths,
            treeSizes = treeSizes,
            featureIds = featureIds.toIntArray(),
            nodeFloatSplits = nodeValues,
            nonLeafValuesCount = nonLeafValuesCount,
            leafValues = targetWeights,
            biases = baseValues ?: FloatArray(numTargets),
            numTargets = numTargets
        )
    }

    companion object {
        private val SUPPORTED_MODES = hashSetOf("LEAF", "BRANCH_GT")

        private const val DEFAULT_AGGREGATOR = "SUM"
        private const val DEFAULT_POST_TRANSFORM = "NONE"

        private fun treeDepthFromNodesNum(n: Int) = ceil(log2(n.toDouble())).toInt()
    }
}
