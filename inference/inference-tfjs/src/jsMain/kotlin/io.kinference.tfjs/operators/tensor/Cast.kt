package io.kinference.tfjs.operators.tensor

import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.extensions.cast
import io.kinference.tfjs.externals.extensions.tidy
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*

class Cast(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(INFO, attributes, inputs, outputs) {

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

    override fun apply(context: Context, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            return@tidy arrayOf(inputs[0]!!.data.cast(tfjsType))
        }

        return listOf(outputs[0].asTensor("output"))
    }
}
