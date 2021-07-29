package io.kinference.operators.logical

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.operators.*
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class Or(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(TensorProto.DataType.BOOL)

        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false))

        private val INFO = OperatorInfo("Or", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val left = inputs[0]!!.data as BooleanNDArray
        val right = inputs[1]!!.data as BooleanNDArray

        val ans = left or right
        return listOf(ans.asTensor("C"))
    }
}
