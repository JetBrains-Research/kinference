package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.LongNDArray
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


    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val array = inputs[0]!!.data as LongNDArray
        val pointer = array.array.pointer()
        val shape =  IntArray(array.linearSize) { pointer.getAndIncrement().toInt() }
        val result = value.data.allocateNDArray(Strides(shape)).apply { fill(value.data.singleValue()) }
        return listOf(result.asTensor("output"))
    }
}
