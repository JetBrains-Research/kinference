package io.kinference.core.operators.ml

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.operators.ml.trees.*
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto.AttributeType
import io.kinference.protobuf.message.TensorProto

sealed class TreeEnsembleRegressor(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in TreeEnsembleRegressorVer1.VERSION.asRange() -> TreeEnsembleRegressorVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of TreeEnsembleRegressor operator: $version")
        }
    }

    @Suppress("PropertyName")
    internal class RegressorInfo(op: Operator<KIONNXData<*>, KIONNXData<*>>) : BaseEnsembleInfo(op) {
        val nTargets: Number = op.getAttribute("n_targets")
        val targetIds: LongArray = op.getAttribute("target_ids")
        val targetNodeIds: LongArray = op.getAttribute("target_nodeids")
        val targetTreeIds: LongArray = op.getAttribute("target_treeids")
        val targetWeights: FloatArray = op.getAttribute("target_weights")
    }
}


class TreeEnsembleRegressorVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : TreeEnsembleRegressor(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            //TensorProto.DataType.INT32,
            //TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        internal val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false)
        )

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

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("TreeEnsembleRegressor", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "ai.onnx.ml")
    }

    private val ensembleInfo: RegressorInfo
        get() = RegressorInfo(this as Operator<KIONNXData<*>, KIONNXData<*>>)

    private val treeEnsemble = TreeEnsembleBuilder.fromInfo(ensembleInfo)

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val inputData = inputs[0]!!.data.toFloatNDArray()
        return listOf(treeEnsemble.execute(inputData).asTensor("Y"))
    }
}
