package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.graph.Context
import io.kinference.onnx.AttributeProto
import io.kinference.onnx.TensorProto.DataType
import io.kinference.operators.AttributeInfo
import io.kinference.operators.IOInfo
import io.kinference.operators.Operator
import io.kinference.operators.OperatorInfo

class Constant(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("value", setOf(AttributeProto.AttributeType.TENSOR), false),
            AttributeInfo("value_float", setOf(AttributeProto.AttributeType.FLOAT), false),
            AttributeInfo("value_floats", setOf(AttributeProto.AttributeType.FLOATS), false),
            AttributeInfo("value_int", setOf(AttributeProto.AttributeType.INT), false),
            AttributeInfo("value_ints", setOf(AttributeProto.AttributeType.INTS), false),
            AttributeInfo("value_string", setOf(AttributeProto.AttributeType.STRING), false),
            AttributeInfo("value_strings", setOf(AttributeProto.AttributeType.STRINGS), false)
            //TODO: sparse tensor values
        )

        private val INPUTS_INFO = emptyList<IOInfo>()

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val INFO = OperatorInfo("Constant", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    @ExperimentalUnsignedTypes
    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        //only one of all attributes is not null
        val (name, value) = ATTRIBUTES_INFO.map { it.name to getAttributeOrNull<Any?>(it.name) }.single { it.second != null }

        @Suppress("UNCHECKED_CAST")
        val result = when (name) {
            "value" -> value
            "value_float" -> Tensor(value!!, DataType.FLOAT)
            "value_floats" -> Tensor(value!! as List<Any>, DataType.FLOAT)
            "value_int" -> Tensor(value!!, DataType.INT64)
            "value_ints" -> Tensor(value!! as List<Any>, DataType.INT64)
            "value_string" -> Tensor(value!!, DataType.STRING)
            "value_strings" -> Tensor(value!! as List<Any>, DataType.STRING)
            else -> error("Unsupported data type")
        } as Tensor
        return listOf(result)
    }
}
