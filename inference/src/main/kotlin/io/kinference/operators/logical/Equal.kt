package io.kinference.operators.logical

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.NDArray
import io.kinference.ndarray.extensions.applyWithBroadcast
import io.kinference.onnx.TensorProto
import io.kinference.operators.IOInfo
import io.kinference.operators.Operator
import io.kinference.operators.OperatorInfo
import io.kinference.primitives.types.DataType

class Equal(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = PRIMITIVE_DATA_TYPES + TensorProto.DataType.BFLOAT16

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.BOOL), "C", optional = false)
        )

        private val INFO = OperatorInfo("Equal", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)

        infix fun NDArray.equal(other: NDArray): NDArray {
            return applyWithBroadcast(other, DataType.BOOLEAN) { first, second, dest ->
                for (i in 0 until dest.linearSize) dest[i] = first[i] == second[i]
            }
        }
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val result = inputs[0]!!.data equal inputs[1]!!.data
        return listOf(result.asTensor("output"))
    }
}
