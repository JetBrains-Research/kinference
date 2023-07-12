package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.trilu
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Trilu(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 14)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Trilu {
            return when(version ?: DEFAULT_VERSION.sinceVersion) {
                in TriluVer14.VERSION.asRange() -> TriluVer14(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Trilu operator: $version")
            }
        }
    }
}


class TriluVer14(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Trilu(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "k", optional = true, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false, differentiable = true)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("upper", setOf(AttributeProto.AttributeType.INT), required = false, default = 1L)
        )

        internal val VERSION = VersionInfo(sinceVersion = 14)
        private val INFO = OperatorInfo("Trilu", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val upper: Boolean by attribute { it: Number -> it != 0L }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        val kTensor = inputs.getOrNull(1)?.data as? NumberNDArrayTFJS

        val k = kTensor?.singleValue()?.toInt() ?: 0
        val output = input.trilu(k, upper)

        return listOf(output.asTensor("output"))
    }
}
