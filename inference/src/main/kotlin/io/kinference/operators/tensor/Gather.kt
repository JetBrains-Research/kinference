package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.extensions.gather
import io.kinference.ndarray.extensions.indexAxis
import io.kinference.onnx.AttributeProto
import io.kinference.onnx.TensorProto
import io.kinference.operators.AttributeInfo
import io.kinference.operators.IOInfo
import io.kinference.operators.Operator
import io.kinference.operators.OperatorInfo

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

    @ExperimentalUnsignedTypes
    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val (data, indices) = inputs
        val axis = data!!.data.indexAxis(axis)
        return listOf(data.data.gather(indices!!.data, axis).asTensor())
    }
}
