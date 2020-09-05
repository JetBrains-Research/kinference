package io.kinference.operators.logical

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.arrays.MutableBooleanNDArray
import io.kinference.onnx.TensorProto
import io.kinference.operators.*

class Not(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(TensorProto.DataType.BOOL)

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val INFO = OperatorInfo("Not", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }

    @ExperimentalUnsignedTypes
    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val data = inputs[0]!!.data.toMutable() as MutableBooleanNDArray
        return listOf(data.not().asTensor("output"))
    }
}
