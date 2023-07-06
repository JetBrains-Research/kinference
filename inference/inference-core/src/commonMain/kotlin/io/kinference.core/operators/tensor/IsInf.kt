package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.DoubleNDArray
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.extensions.isInf.isInf
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto.AttributeType
import io.kinference.protobuf.message.TensorProto

sealed class IsInf(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
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
    inputs: List<String>,
    outputs: List<String>
) : IsInf(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("detect_negative", setOf(AttributeType.INT), default = 1),
            AttributeInfo("detect_positive", setOf(AttributeType.INT), default = 1),
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

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data

        val output =  when (input.type) {
            DataType.FLOAT -> (input as FloatNDArray).isInf(detectNegative, detectPositive)
            DataType.DOUBLE -> (input as DoubleNDArray).isInf(detectNegative, detectPositive)
            else -> error("")
        }

        return listOf(output.asTensor("Y"))
    }
}


