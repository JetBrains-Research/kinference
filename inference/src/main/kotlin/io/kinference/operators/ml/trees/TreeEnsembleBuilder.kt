package io.kinference.operators.ml.trees

import io.kinference.ndarray.toIntArray
import io.kinference.operators.ml.*
import io.kinference.protobuf.message.TensorProto
import kotlin.math.ceil
import kotlin.math.log2

internal class TreeEnsembleBuilder(private val info: TreeEnsembleOperator.BaseEnsembleInfo, private val labelsInfo: TreeEnsemble.LabelsInfo?) {
    private val numTargets = labelsInfo?.labels?.size ?: 1
    private val numTrees: Int = info.nodes_treeids.distinct().size
    private val treeDepths: IntArray = IntArray(numTrees)
    private val treeSizes: IntArray = IntArray(numTrees)
    private var biases: FloatArray = info.base_values ?: FloatArray(numTargets)
    private val nonLeafValuesCount: IntArray = IntArray(numTrees)
    private lateinit var weightValues: FloatArray

    init {
        require(numTargets > 0) { "Number of targets should be > 0, got $numTargets" }

        val trees2nodes = info.nodes_treeids.zip(info.nodes_nodeids).groupBy { it.first }
        for (entry in trees2nodes) {
            val treeId = entry.key.toInt()
            val currentNumNodes = entry.value.size
            treeSizes[treeId] = currentNumNodes
            treeDepths[treeId] = DEPTH(currentNumNodes)
        }

        for (entry in trees2nodes) {
            val treeId = entry.key.toInt()
            val treeOffset = treeSizes.take(treeId).fold(0, Int::plus)
            for (tree2node in entry.value) {
                val nodeOffset = tree2node.second.toInt() + treeOffset
                if (info.nodes_modes[nodeOffset] != "LEAF") nonLeafValuesCount[treeId]++
            }
        }
    }

    fun withWeights(targetIds: LongArray, targetNodeIdsList: LongArray, targetTreeIds: LongArray, targetWeights: FloatArray) {
        assert(targetNodeIdsList.size == targetTreeIds.size)
        assert(targetNodeIdsList.size == targetWeights.size)
        assert(!targetIds.asSequence().chunked(numTargets).any { !checkOrder(it) })

        weightValues = targetWeights
    }

    private fun checkNodeDependencies(trueNodeIds: LongArray, falseNodeIds: LongArray) {
        assert(trueNodeIds.size == falseNodeIds.size)
        assert(trueNodeIds.size == info.nodes_values.size)

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

    fun build(): TreeEnsemble {
        val aggregatorName = if (info.aggregate_function in AGGREGATIONS) info.aggregate_function!! else DEFAULT_AGGREGATOR
        val postTransformName = if (info.post_transform in TRANSFORMATIONS) info.post_transform!! else DEFAULT_POST_TRANSFORM
        val aggregator = Aggregator[aggregatorName]
        val postTransform = PostTransform[postTransformName]

        checkNodeDependencies(info.nodes_truenodeids, info.nodes_falsenodeids)
        return TreeEnsemble(
            aggregator, postTransform, treeDepths, treeSizes, info.nodes_featureids.toIntArray(),
            info.nodes_values, nonLeafValuesCount, weightValues, biases, numTargets, labelsInfo
        )
    }

    companion object {
        operator fun invoke(
            info: TreeEnsembleOperator.BaseEnsembleInfo,
            labelsInfo: TreeEnsemble.LabelsInfo? = null,
            configure: TreeEnsembleBuilder.() -> Unit
        ): TreeEnsembleBuilder = TreeEnsembleBuilder(info, labelsInfo).apply(configure)

        private fun checkOrder(list: List<Number>): Boolean {
            for (i in list.indices) {
                if (list[i].toInt() != i) return false
            }
            return true
        }

        private fun checkInfo(info: TreeEnsembleOperator.BaseEnsembleInfo) {
            val distinctModes = info.nodes_modes.toHashSet()
            require(distinctModes == SUPPORTED_MODES) { "Got unsupported tree ensemble node modes: ${distinctModes - SUPPORTED_MODES}" }

            assert(info.nodes_nodeids.size == info.nodes_modes.size)
            assert(info.nodes_nodeids.size == info.nodes_treeids.size)
            assert(info.nodes_nodeids.size == info.nodes_values.size)
            assert(info.nodes_nodeids.size == info.nodes_featureids.size)
        }

        internal fun fromInfo(info: TreeEnsembleRegressor.RegressorInfo): TreeEnsemble {
            checkInfo(info)
            return TreeEnsembleBuilder(info) {
                withWeights(info.target_ids, info.target_nodeids, info.target_treeids, info.target_weights)
            }.build()
        }

        private fun parseLabelsInfo(info: TreeEnsembleClassifier.ClassifierInfo): TreeEnsemble.LabelsInfo {
            return when {
                info.classlabels_int64s != null -> TreeEnsemble.LabelsInfo(info.classlabels_int64s!!.toList(), TensorProto.DataType.INT64)
                info.classlabels_strings != null -> TreeEnsemble.LabelsInfo(info.classlabels_strings!!, TensorProto.DataType.STRING)
                else -> error("Either classlabels_int64s or classlabels_strings should be present")
            }
        }

        internal fun fromInfo(info: TreeEnsembleClassifier.ClassifierInfo): TreeEnsemble {
            checkInfo(info)
            val labels = parseLabelsInfo(info)
            return TreeEnsembleBuilder(info, labels) {
                withWeights(info.class_ids, info.class_nodeids, info.class_treeids, info.class_weights)
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
