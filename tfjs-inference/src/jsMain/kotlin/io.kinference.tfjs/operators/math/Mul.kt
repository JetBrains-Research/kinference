package io.kinference.tfjs.operators.math

import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.custom_externals.extensions.tidy
import io.kinference.tfjs.custom_externals.extensions.times
import io.kinference.tfjs.data.tensors.Tensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*
import io.kinference.protobuf.message.TensorProto

class Mul(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
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

        private val INFO = OperatorInfo("Mul", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val result = tidy { arrayOf(inputs[0]!!.data * inputs[1]!!.data) }.first()
        return listOf(result.asTensor("C"))
    }
}

