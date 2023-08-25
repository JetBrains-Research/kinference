package io.kinference.tfjs.operators.layer.normalization

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.batchNorm
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class BatchNormalization(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): BatchNormalization {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in BatchNormalizationVer9.VERSION.asRange() -> BatchNormalizationVer9(name, attributes, inputs, outputs)
                else -> error("Unsupported version of BatchNormalization operator: $version")
            }
        }
    }
}

class BatchNormalizationVer9 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : BatchNormalization(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "scale", optional = false),
            IOInfo(2, TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(3, TYPE_CONSTRAINTS, "input_mean", optional = false),
            IOInfo(4, TYPE_CONSTRAINTS, "input_var", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), required = false, default = 1e-05),
            AttributeInfo("momentum", setOf(AttributeProto.AttributeType.FLOAT), required = false, default = 0.9),
            AttributeInfo("training_mode", setOf(AttributeProto.AttributeType.INT), required = false, default = 0L),
        )

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("BatchNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.DEFAULT_DOMAIN)
    }

    private val epsilon: Float by attribute { it: Number -> it.toFloat() }
    private val momentum: Number by attribute()
    private val trainingMode: Boolean by attribute("training_mode") { it: Number -> it.toInt() != 0 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        require(!trainingMode) { "Training mode is not supported for BatchNormalization operator" }

        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val scale = inputs[1]!!.data as NumberNDArrayTFJS
        val bias = inputs[2]!!.data as NumberNDArrayTFJS
        val mean = inputs[3]!!.data as NumberNDArrayTFJS
        val variance = inputs[4]!!.data as NumberNDArrayTFJS

        val output = input.batchNorm(scale, bias, mean, variance, epsilon)
        return listOf(output.asTensor("Y"))
    }
}
