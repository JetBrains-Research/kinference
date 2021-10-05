package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.custom_externals.core.broadcastTo
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import org.khronos.webgl.Int32Array
import io.kinference.custom_externals.core.tensor
import io.kinference.custom_externals.extensions.*
import io.kinference.graph.Context


class ConstantOfShape(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {

    companion object {
        private val TYPE_CONSTRAINTS = PRIMITIVE_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("value", setOf(AttributeProto.AttributeType.TENSOR),
                default = tensor(floatArrayOf(0f), arrayOf(1), "float32").asTensor("value"), required = false)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val INFO = OperatorInfo("ConstantOfShape", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val value: Tensor by attribute()

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            val shape = inputs[0]!!.data.dataInt().toTypedArray()
            if (shape.contains(0)) {
                return@tidy arrayOf(tensor(emptyArray(), shape, value.data.dtype))
            }

            val output = value.data.broadcastTo(shape)
            return@tidy arrayOf(output)
        }

        return listOf(outputs[0].asTensor("output"))
    }
}

