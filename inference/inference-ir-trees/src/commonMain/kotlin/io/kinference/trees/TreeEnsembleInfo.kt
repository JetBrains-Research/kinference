package io.kinference.trees

import io.kinference.utils.toIntArray

data class TreeEnsembleInfo(
    val aggregator: Aggregator,
    val transformType: PostTransformType,
    val treeSizes: IntArray,
    val featureIds: IntArray,
    val nodeFloatSplits: FloatArray,
    val nextNodeIds: IntArray,
    val leafValues: FloatArray,
    val leafCounter: IntArray,
    val biases: FloatArray,
    val numTargets: Int,
    val splitMode: TreeSplitType
) {
    companion object {
        private val SUPPORTED_MODES = hashSetOf("LEAF", "BRANCH_GT", "BRANCH_GTE", "BRANCH_LT", "BRANCH_LEQ")

        private const val DEFAULT_AGGREGATOR = "SUM"
        private const val DEFAULT_POST_TRANSFORM = "NONE"

        //private fun treeDepthFromNodesNum(n: Int) = ceil(log2(n.toDouble())).toInt()

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
            val treeSizes = IntArray(numTrees)

            //count the number of nodes and compute the depth of each tree
            val trees2nodes = treeIds.zip(nodeIds).groupBy { it.first }
            for (entry in trees2nodes) {
                val treeId = entry.key.toInt()
                val currentNumNodes = entry.value.size
                treeSizes[treeId] = currentNumNodes
            }

            val prevLeavesCounter = IntArray(nodeIds.size)
            var counter = 0; var treeOffset = 0

            for (treeSize in treeSizes) {
                for (i in 0 until treeSize) {
                    val currentIdx = treeOffset + i
                    prevLeavesCounter[currentIdx] = counter
                    if (nodeModes[currentIdx] == "LEAF") counter++
                }
                treeOffset += treeSize
            }

            val nextNodeIds = IntArray(falseNodeIds.size + trueNodeIds.size)
            for (i in falseNodeIds.indices) {
                nextNodeIds[2 * i] = falseNodeIds[i].toInt()
                nextNodeIds[2 * i + 1] = trueNodeIds[i].toInt()
            }

            return TreeEnsembleInfo(
                aggregator = Aggregator[aggregatorType],
                transformType = postTransformType,
                treeSizes = treeSizes,
                featureIds = featureIds.toIntArray(),
                nodeFloatSplits = nodeValues,
                nextNodeIds = nextNodeIds,
                leafValues = targetWeights,
                leafCounter = prevLeavesCounter,
                biases = baseValues ?: FloatArray(numTargets),
                numTargets = numTargets,
                splitMode = TreeSplitType.valueOf(splitMode)
            )
        }
    }
}
