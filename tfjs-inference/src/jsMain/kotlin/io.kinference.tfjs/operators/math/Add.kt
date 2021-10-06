package io.kinference.tfjs.operators.math

import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.custom_externals.extensions.plus
import io.kinference.tfjs.custom_externals.extensions.tidy
import io.kinference.tfjs.data.tensors.Tensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*
import io.kinference.protobuf.message.TensorProto

class Add(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false)
        )

        private val INFO = OperatorInfo("Add", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }


    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            val left = inputs[0]!!
            val right = inputs[1]!!
            return@tidy arrayOf(left.data + right.data)
        }
        return listOf(outputs[0].asTensor("C"))
    }
}
