package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.custom_externals.extensions.*
import io.kinference.tfjs.data.tensors.Tensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

class Gather(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT32, TensorProto.DataType.INT64), "indices", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val INFO = OperatorInfo("Gather", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }


    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val outputs = tidy {
            val data = inputs[0]!!.data
            val indices = inputs[1]!!.data
            val actualAxis = data.indexAxis(axis)
            val dim = data.shape[actualAxis]

            val indicesData = indices.dataInt()
            for (idx in indicesData.indices) {
                val value = indicesData[idx]
                if (value < 0) indicesData[idx] = value + dim
            }
            val preparedIndices = tensor(indicesData, indices.shape, indices.dtype)

            val output = data.gather(preparedIndices, actualAxis)

            return@tidy arrayOf(output)
        }
        return listOf(outputs[0].asTensor("output"))
    }
}

