package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.asTensor

sealed class Div(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 7)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): DivVer7 {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in DivVer7.VERSION.asRange() -> DivVer7(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Div operator: $version")
            }
        }
    }
}


class DivVer7(
    name: String,
    attributes:
    Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Div(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.UINT16,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT8,
            TensorProto.DataType.INT16,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", differentiable = true, optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", differentiable = true, optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", differentiable = true, optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("Div", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val first = inputs[0]!!.data as NumberNDArrayTFJS
        val second = inputs[1]!!.data as NumberNDArrayTFJS
        val result = first / second

        return listOf(result.asTensor("C"))
    }
}

