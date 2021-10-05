package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.custom_externals.core.cast
import io.kinference.custom_externals.extensions.cast
import io.kinference.custom_externals.extensions.tidy
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

class Cast(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {

    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("to", setOf(AttributeProto.AttributeType.INT), true)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val INFO = OperatorInfo("Cast", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val toType: Int by attribute("to") { it: Long -> it.toInt() }

    private val tfjsType = when(TensorProto.DataType.fromValue(toType)) {
        TensorProto.DataType.INT64, TensorProto.DataType.UINT64,
        TensorProto.DataType.INT32, TensorProto.DataType.UINT32,
        TensorProto.DataType.INT16, TensorProto.DataType.UINT16,
        TensorProto.DataType.INT8, TensorProto.DataType.UINT8 -> "int32"

        TensorProto.DataType.FLOAT, TensorProto.DataType.DOUBLE, TensorProto.DataType.BFLOAT16 -> "float32"

        TensorProto.DataType.BOOL -> "bool"

        TensorProto.DataType.COMPLEX64, TensorProto.DataType.COMPLEX128 -> "complex64"
        TensorProto.DataType.STRING -> "string"
        else -> error("Unsupported type")
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            return@tidy arrayOf(inputs[0]!!.data.cast(tfjsType))
        }

        return listOf(outputs[0].asTensor("output"))
    }
}
