package io.kinference.core.operators.ml

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.core.operators.*
import io.kinference.core.operators.ml.TreeEnsembleOperator.Companion.toFloatNDArray
import io.kinference.core.operators.ml.trees.TreeEnsembleBuilder
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto.AttributeType
import io.kinference.protobuf.message.TensorProto

@ExperimentalTime
class TreeEnsembleRegressor(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val OUTPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.FLOAT), "Y", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("aggregate_function", setOf(AttributeType.STRING), required = false, default = "SUM"),
            AttributeInfo("base_values", setOf(AttributeType.FLOATS), required = false),
            AttributeInfo("n_targets", setOf(AttributeType.INT), required = true),
            AttributeInfo("nodes_falsenodeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_featureids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_hitrates", setOf(AttributeType.FLOATS), required = false),
            AttributeInfo("nodes_missing_value_tracks_true", setOf(AttributeType.INTS), required = false),
            AttributeInfo("nodes_modes", setOf(AttributeType.STRINGS), required = true),
            AttributeInfo("nodes_nodeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_treeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_truenodeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_values", setOf(AttributeType.FLOATS), required = true),
            AttributeInfo("post_transform", setOf(AttributeType.STRING), required = false, default = "NONE"),
            AttributeInfo("target_ids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("target_nodeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("target_treeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("target_weights", setOf(AttributeType.FLOATS), required = true)
        )

        private val INFO = OperatorInfo("TreeEnsembleRegressor", ATTRIBUTES_INFO, TreeEnsembleOperator.INPUTS_INFO, OUTPUTS_INFO)
    }

    @Suppress("PropertyName")
    class RegressorInfo(map: Map<String, Any?>) : TreeEnsembleOperator.BaseEnsembleInfo(map) {
        val n_targets: Number by map
        val target_ids: LongArray by map
        val target_nodeids: LongArray by map
        val target_treeids: LongArray by map
        val target_weights: FloatArray by map
    }

    private val ensembleInfo: RegressorInfo
        get() = RegressorInfo(this.attributes.mapValues { it.value.value }.withDefault { null })

    private val treeEnsemble = TreeEnsembleBuilder.fromInfo(ensembleInfo)

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val inputData = inputs[0]!!.data.toFloatNDArray()
        return listOf(treeEnsemble.execute(inputData).asTensor("Y"))
    }
}
