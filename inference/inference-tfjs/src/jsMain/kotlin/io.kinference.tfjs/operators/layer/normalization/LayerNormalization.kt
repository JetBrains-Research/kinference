package io.kinference.tfjs.operators.layer.normalization

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class LayerNormalization(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in LayerNormalizationVer1.VERSION.asRange() -> LayerNormalizationVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of LayerNormalization operator: $version")
            }
    }
}

class LayerNormalizationVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    LayerNormalization(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, -1),
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), false, 0.00001f)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", false),
            IOInfo(1, TYPE_CONSTRAINTS, "scale", false),
            IOInfo(2, TYPE_CONSTRAINTS, "B", false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", false),
            //Unsupported
            IOInfo(1, TYPE_CONSTRAINTS, "mean", true),
            IOInfo(2, TYPE_CONSTRAINTS, "inv_std_var", true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("LayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }
    private val epsilon: Float by attribute()

    private val epsilonScalar = scalar(epsilon, "float32")

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val scale = inputs[1]!!.data as NumberNDArrayTFJS
        val bias = inputs[2]!!.data as NumberNDArrayTFJS

        val actualAxis = input.indexAxis(axis)
        val axesForMoments = Array(input.rank - actualAxis) { actualAxis + it }

        val (mean, variance) = input.moments(axesForMoments, keepDims = true)

        val epsilonTensor = NumberNDArrayTFJS(epsilonScalar)
        val normalizedInput = (input - mean) / (variance + epsilonTensor).tfjs { it.sqrt() } * scale + bias

        return listOf(normalizedInput.asTensor("Y")).also {
            closeAll(mean, variance)
        }
    }
}

