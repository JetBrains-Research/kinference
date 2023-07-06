package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.isInf
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class IsInf(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in IsInfVer10.VERSION.asRange() -> IsInfVer10(name, attributes, inputs, outputs)
                else -> error("Unsupported version of IsInf operator: $version")
            }
    }
}

class IsInfVer10(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : IsInf(name, INFO, attributes, inputs, outputs) {

    companion object {
        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("detect_negative", setOf(AttributeProto.AttributeType.INT), default = 1),
            AttributeInfo("detect_positive", setOf(AttributeProto.AttributeType.INT), default = 1),
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.DOUBLE), "X", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.BOOL), "Y", optional = false)
        )

        //Realized the latest version, but there is backward compatibility between operators
        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("IsInf", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val detectNegative: Boolean by attribute("detect_negative") { it: Number -> it.toInt() == 1 }
    private val detectPositive: Boolean by attribute("detect_positive") { it: Number -> it.toInt() == 1 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        return listOf(input.isInf(detectNegative, detectPositive).asTensor("Y"))
    }
}
