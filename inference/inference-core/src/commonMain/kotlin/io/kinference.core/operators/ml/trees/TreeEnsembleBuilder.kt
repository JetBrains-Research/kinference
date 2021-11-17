package io.kinference.core.operators.ml.trees

import io.kinference.core.KIONNXData
import io.kinference.core.operators.Operator
import io.kinference.ndarray.toIntArray
import io.kinference.core.operators.ml.*
import kotlin.math.ceil
import kotlin.math.log2
import kotlin.time.ExperimentalTime

internal open class BaseEnsembleInfo(op: Operator<KIONNXData<*>, KIONNXData<*>>) {
    init {
        require(op.info.name == "TreeEnsembleClassifier" || op.info.name == "TreeEnsembleRegressor")
    }

    val aggregateFunc: String? = op.getAttributeOrNull("aggregate_function")
    val baseValues: FloatArray? = op.getAttributeOrNull("base_values")
    val falseNodeIds: LongArray = op.getAttribute("nodes_falsenodeids")
    val featureIds: LongArray = op.getAttribute("nodes_featureids")
    val nodeModes: List<String> = op.getAttribute("nodes_modes")
    val nodeIds: LongArray = op.getAttribute("nodes_nodeids")
    val treeIds: LongArray = op.getAttribute("nodes_treeids")
    val trueNodeIds: LongArray = op.getAttribute("nodes_truenodeids")
    val nodeValues: FloatArray = op.getAttribute("nodes_values")
    val postTransform: String? = op.getAttributeOrNull("post_transform")
}

@ExperimentalTime
internal class TreeEnsembleBuilder(private val info: BaseEnsembleInfo, private val labelsInfo: TreeEnsemble.LabelsInfo<*>?) {
    private val numTargets = labelsInfo?.labels?.size ?: 1
    private val numTrees: Int = info.treeIds.distinct().size
    private val treeDepths: IntArray = IntArray(numTrees)
    private val treeSizes: IntArray = IntArray(numTrees)
    private var biases: FloatArray = info.baseValues ?: FloatArray(numTargets)
    private val nonLeafValuesCount: IntArray = IntArray(numTrees)
    private lateinit var weightValues: FloatArray

    init {
        require(numTargets > 0) { "Number of targets should be > 0, got $numTargets" }

        val trees2nodes = info.treeIds.zip(info.nodeIds).groupBy { it.first }
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
                if (info.nodeModes[nodeOffset] != "LEAF") nonLeafValuesCount[treeId]++
            }
        }
    }

    fun withWeights(targetIds: LongArray, targetNodeIdsList: LongArray, targetTreeIds: LongArray, targetWeights: FloatArray) {
        require(targetNodeIdsList.size == targetTreeIds.size)
        require(targetNodeIdsList.size == targetWeights.size)
        require(!targetIds.asSequence().chunked(numTargets).any { !checkOrder(it) })

        weightValues = targetWeights
    }

    private fun checkNodeDependencies(trueNodeIds: LongArray, falseNodeIds: LongArray) {
        require(trueNodeIds.size == falseNodeIds.size)
        require(trueNodeIds.size == info.nodeValues.size)

        var currentTreeOffset = 0
        for (size in treeSizes) {
            for (j in currentTreeOffset until currentTreeOffset + size) {
                val falseId = falseNodeIds[j].toInt()
                val trueId = trueNodeIds[j].toInt()
                if (info.nodeModes[j] != "LEAF" && (2 * info.nodeIds[j].toInt() + 1 != falseId || 2 * info.nodeIds[j].toInt() + 2 != trueId)) {
                    error("Incorrect tree nodes order")
                }
            }
            currentTreeOffset += size
        }
    }

    fun build(): TreeEnsemble {
        val aggregatorName = if (info.aggregateFunc in AGGREGATIONS) info.aggregateFunc!! else DEFAULT_AGGREGATOR
        val postTransformName = if (info.postTransform in TRANSFORMATIONS) info.postTransform!! else DEFAULT_POST_TRANSFORM
        val aggregator = Aggregator[aggregatorName]
        val postTransform = PostTransform[postTransformName]

        checkNodeDependencies(info.trueNodeIds, info.falseNodeIds)
        return TreeEnsemble(
            aggregator, postTransform, treeDepths, treeSizes, info.featureIds.toIntArray(),
            info.nodeValues, nonLeafValuesCount, weightValues, biases, numTargets, labelsInfo
        )
    }

    companion object {
        operator fun invoke(
            info: BaseEnsembleInfo,
            labelsInfo: TreeEnsemble.LabelsInfo<*>? = null,
            configure: TreeEnsembleBuilder.() -> Unit
        ): TreeEnsembleBuilder = TreeEnsembleBuilder(info, labelsInfo).apply(configure)

        private fun checkOrder(list: List<Number>): Boolean {
            for (i in list.indices) {
                if (list[i].toInt() != i) return false
            }
            return true
        }

        private fun checkInfo(info: BaseEnsembleInfo) {
            val distinctModes = info.nodeModes.toHashSet()
            require(distinctModes == SUPPORTED_MODES) { "Got unsupported tree ensemble node modes: ${distinctModes - SUPPORTED_MODES}" }

            require(info.nodeIds.size == info.nodeModes.size)
            require(info.nodeIds.size == info.treeIds.size)
            require(info.nodeIds.size == info.nodeValues.size)
            require(info.nodeIds.size == info.featureIds.size)
        }

        internal fun fromInfo(info: TreeEnsembleRegressor.RegressorInfo): TreeEnsemble {
            checkInfo(info)
            return TreeEnsembleBuilder(info) {
                withWeights(info.targetIds, info.targetNodeIds, info.targetTreeIds, info.targetWeights)
            }.build()
        }

        internal fun fromInfo(info: TreeEnsembleClassifier.ClassifierInfo): TreeEnsemble {
            checkInfo(info)
            return TreeEnsembleBuilder(info, info.classLabels) {
                withWeights(info.classIds, info.classNodeIds, info.classTreeIds, info.classWeights)
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
