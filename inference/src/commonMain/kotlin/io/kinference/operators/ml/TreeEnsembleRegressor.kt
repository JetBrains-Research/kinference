package io.kinference.operators.ml

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.onnx.AttributeProto.AttributeType
import io.kinference.onnx.TensorProto
import io.kinference.operators.*
import io.kinference.operators.ml.TreeEnsembleOperator.Companion.toFloatNDArray
import io.kinference.operators.ml.trees.TreeEnsembleBuilder
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TreeEnsembleRegressor(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
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
        val target_ids: List<Number> by map
        val target_nodeids: List<Number> by map
        val target_treeids: List<Number> by map
        val target_weights: List<Number> by map
    }

    private val ensembleInfo: RegressorInfo
        get() = RegressorInfo(this.attributes.mapValues { it.value.value }.withDefault { null })

    private val treeEnsemble = TreeEnsembleBuilder.fromInfo(ensembleInfo)

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val inputData = inputs[0]!!.data.toFloatNDArray()
        return listOf(treeEnsemble.execute(inputData).asTensor("Y"))
    }
}
