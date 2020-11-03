package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.*
import io.kinference.graph.Context
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.onnx.AttributeProto
import io.kinference.onnx.TensorProto
import io.kinference.operators.*
import io.kinference.types.TensorInfo
import io.kinference.types.TensorShape

class ConstantOfShape(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = PRIMITIVE_DATA_TYPES

        private val DEFAULT_TENSOR = Tensor(FloatNDArray(floatArrayOf(0.0f)), TensorInfo("value", TensorProto.DataType.FLOAT, TensorShape(IntArray(0))))
        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("value", setOf(AttributeProto.AttributeType.TENSOR), default = DEFAULT_TENSOR, required = false)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val INFO = OperatorInfo("ConstantOfShape", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val value: Tensor by attribute()

    @ExperimentalUnsignedTypes
    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        @Suppress("UNCHECKED_CAST")
        val shape = inputs[0]!!.data.let { IntArray(it.linearSize) { i -> (it[i] as Number).toInt() } }
        val result = value.data.allocateNDArray(Strides(shape)).apply { fill(value.data[0]) }
        return listOf(result.asTensor("output"))
    }
}
