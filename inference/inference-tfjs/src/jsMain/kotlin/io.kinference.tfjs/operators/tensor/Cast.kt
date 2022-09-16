package io.kinference.tfjs.operators.tensor

import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.ndarray.extensions.cast

sealed class Cast(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in CastVer6.VERSION.asRange() -> CastVer6(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Cast operator: $version")
        }
    }
}

class CastVer6(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Cast(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("to", setOf(AttributeProto.AttributeType.INT), true)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Cast", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val toType: Int by attribute("to") { it: Long -> it.toInt() }

    private val tfjsType = when(val type = TensorProto.DataType.fromValue(toType)) {
        TensorProto.DataType.INT64, TensorProto.DataType.UINT64,
        TensorProto.DataType.INT32, TensorProto.DataType.UINT32,
        TensorProto.DataType.INT16, TensorProto.DataType.UINT16,
        TensorProto.DataType.INT8, TensorProto.DataType.UINT8 -> "int32"

        TensorProto.DataType.FLOAT, TensorProto.DataType.DOUBLE, TensorProto.DataType.BFLOAT16 -> "float32"

        TensorProto.DataType.BOOL -> "bool"

        TensorProto.DataType.COMPLEX64, TensorProto.DataType.COMPLEX128 -> "complex64"
        TensorProto.DataType.STRING -> "string"
        else -> error("Unsupported type: $type")
    }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        return listOf(input.cast(tfjsType).asTensor("output"))
    }
}
