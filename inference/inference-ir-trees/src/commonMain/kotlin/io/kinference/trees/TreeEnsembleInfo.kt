package io.kinference.trees

import kotlin.math.ceil
import kotlin.math.log2

data class TreeEnsembleInfo(
    val aggregator: Aggregator,
    val transform: PostTransform,
    val treeDepths: IntArray,
    val treeSizes: IntArray,
    val featureIds: IntArray,
    val nodeFloatSplits: FloatArray,
    val nonLeafValuesCount: IntArray,
    val leafValues: FloatArray,
    val biases: FloatArray,
    val numTargets: Int,
    val splitMode: TreeSplitType
) {
    companion object {
        private val SUPPORTED_MODES = hashSetOf("LEAF", "BRANCH_GT", "BRANCH_GTE", "BRANCH_LT", "BRANCH_LEQ")

        private const val DEFAULT_AGGREGATOR = "SUM"
        private const val DEFAULT_POST_TRANSFORM = "NONE"

        private fun treeDepthFromNodesNum(n: Int) = ceil(log2(n.toDouble())).toInt()

        operator fun invoke(
            baseValues: FloatArray?, featureIds: LongArray, nodeModes: List<String>, nodeIds: LongArray,
            treeIds: LongArray, falseNodeIds: LongArray, trueNodeIds: LongArray, nodeValues: FloatArray,
            postTransform: String?, aggregator: String?, targetIds: LongArray, targetNodeIds: LongArray,
            targetNodeTreeIds: LongArray, targetWeights: FloatArray, numTargets: Int = 1
        ): TreeEnsembleInfo {
            require(numTargets > 0) { "Number of targets should be > 0, got $numTargets" }

            val postTransformType = PostTransformType.valueOf(postTransform ?: DEFAULT_POST_TRANSFORM)
            val aggregatorType = AggregatorType.valueOf(aggregator ?: DEFAULT_AGGREGATOR)

            val distinctModes = nodeModes.toHashSet()
            require(SUPPORTED_MODES.containsAll(distinctModes)) { "Got unsupported tree ensemble node modes: ${distinctModes - SUPPORTED_MODES}" }
            require(distinctModes.size == 2) { "Only single split mode trees are supported" }

            val splitMode = (distinctModes - "LEAF").single()

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

            return TreeEnsembleInfo(
                aggregator = Aggregator[aggregatorType],
                transform = PostTransform[postTransformType],
                treeDepths = treeDepths,
                treeSizes = treeSizes,
                featureIds = IntArray(featureIds.size) { featureIds[it].toInt() },
                nodeFloatSplits = nodeValues,
                nonLeafValuesCount = nonLeafValuesCount,
                leafValues = targetWeights,
                biases = baseValues ?: FloatArray(numTargets),
                numTargets = numTargets,
                splitMode = TreeSplitType.valueOf(splitMode)
            )
        }
    }
}
