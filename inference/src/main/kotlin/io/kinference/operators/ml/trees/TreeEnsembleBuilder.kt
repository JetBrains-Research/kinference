package io.kinference.operators.ml.trees

import io.kinference.ndarray.toFloatArray
import io.kinference.ndarray.toIntArray
import io.kinference.operators.ml.*
import kotlin.math.*

internal class TreeEnsembleBuilder(val numTargets: Int, val info: TreeEnsembleOperator.BaseEnsembleInfo) {
    private val numTrees: Int = info.nodes_treeids.distinct().size
    private val nodeValues: FloatArray = info.nodes_values.toFloatArray()
    private val featureIds: IntArray = info.nodes_featureids.toIntArray()
    private val treeDepths: IntArray = IntArray(numTrees)
    private val treeSizes: IntArray = IntArray(numTrees)
    private var biases: FloatArray? = info.base_values?.toFloatArray()
    private lateinit var leafValues: FloatArray

    init {
        val trees2nodes = info.nodes_treeids.zip(info.nodes_nodeids).groupBy { it.first }
        for (entry in trees2nodes) {
            val treeId = entry.key.toInt()
            val currentNumNodes = entry.value.size
            treeSizes[treeId] = currentNumNodes
            treeDepths[treeId] = DEPTH(currentNumNodes)
        }
    }

    init {
        require(numTargets > 0) { "Number of targets should be > 0, got $numTargets" }
    }


    //TODO: build ensemble with multi-weight nodes (for TreeEnsembleClassifier)
    fun withWeights(targetIds: List<Number>, targetNodeIds: List<Number>, targetTreeIds: List<Number>, targetWeights: List<Number>) {
        assert(targetNodeIds.size == targetTreeIds.size)
        assert(targetNodeIds.size == targetWeights.size)

        leafValues = FloatArray(nodeValues.size)

        for (i in targetNodeIds.indices) {
            val treeOff = treeSizes.take(targetTreeIds[i].toInt()).fold(0, Int::plus)
            leafValues[treeOff + targetNodeIds[i].toInt()] = targetWeights[i].toFloat()
        }
    }

    private fun checkNodeDependencies(trueNodeIds: List<Number>, falseNodeIds: List<Number>) {
        assert(trueNodeIds.size == falseNodeIds.size)
        assert(trueNodeIds.size == nodeValues.size)

        var currentTreeOffset = 0
        for (size in treeSizes) {
            for (j in currentTreeOffset until currentTreeOffset + size) {
                val falseId = falseNodeIds[j].toInt()
                val trueId = trueNodeIds[j].toInt()
                if (info.nodes_modes[j] != "LEAF" && (2 * info.nodes_nodeids[j].toInt() + 1 != falseId || 2 * info.nodes_nodeids[j].toInt() + 2 != trueId)) {
                    error("Incorrect tree nodes order")
                }
            }
            currentTreeOffset += size
        }
    }

    fun build(): AbstractTreeEnsemble {
        val aggregatorName = if (info.aggregate_function in AGGREGATIONS) info.aggregate_function!! else DEFAULT_AGGREGATOR
        val postTransformName = if (info.post_transform in TRANSFORMATIONS) info.post_transform!! else DEFAULT_POST_TRANSFORM
        val aggregator = Aggregator[aggregatorName]
        val postTransform = PostTransform[postTransformName]

        checkNodeDependencies(info.nodes_truenodeids, info.nodes_falsenodeids)
        return when {
            numTargets == 1 -> SingleTargetEnsemble(aggregator, postTransform, treeDepths, treeSizes, featureIds, nodeValues, leafValues, biases ?: floatArrayOf(0f))
            numTargets >= 1 -> error("Multi-target ensembles are not supported yet")
            else -> error("Number of targets should be > 0, got $numTargets")
        }
    }

    companion object {
        operator fun invoke(numTargets: Int, info: TreeEnsembleOperator.BaseEnsembleInfo, configure: TreeEnsembleBuilder.() -> Unit): TreeEnsembleBuilder {
            return TreeEnsembleBuilder(numTargets, info).apply(configure)
        }

        private fun checkInfo(info: TreeEnsembleOperator.BaseEnsembleInfo) {
            val distinctModes = info.nodes_modes.toHashSet()
            require(distinctModes == SUPPORTED_MODES) { "Got unsupported tree ensemble node modes: ${distinctModes - SUPPORTED_MODES}" }

            assert(info.nodes_nodeids.size == info.nodes_modes.size)
            assert(info.nodes_nodeids.size == info.nodes_treeids.size)
            assert(info.nodes_nodeids.size == info.nodes_values.size)
            assert(info.nodes_nodeids.size == info.nodes_featureids.size)
        }

        internal fun fromInfo(info: TreeEnsembleRegressor.RegressorInfo): AbstractTreeEnsemble {
            checkInfo(info)
            return TreeEnsembleBuilder(info.n_targets.toInt(), info) {
                withWeights(info.target_ids, info.target_nodeids, info.target_treeids, info.target_weights)
            }.build()
        }

        //actually, works correctly only if the tree is complete
        val DEPTH = { n: Int -> ceil(log2(n.toDouble())).toInt() }

        private val SUPPORTED_MODES = hashSetOf("LEAF", "BRANCH_GT")
        private val AGGREGATIONS = hashSetOf("SUM", "MIN", "MAX", "AVERAGE")
        private const val DEFAULT_AGGREGATOR = "SUM"

        private val TRANSFORMATIONS = hashSetOf("NONE", "SOFTMAX", "SOFTMAX_ZERO", "LOGISTIC", "PROBIT")
        private const val DEFAULT_POST_TRANSFORM = "NONE"
    }
}
