package io.kinference.tfjs.operators.math

import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.core.NDArrayTFJS
import io.kinference.tfjs.externals.core.reshape
import io.kinference.tfjs.externals.extensions.matMul
import io.kinference.tfjs.externals.extensions.tidy
import io.kinference.tfjs.graph.Context
import io.kinference.tfjs.operators.*

class MatMul(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(INFO, attributes, inputs, outputs) {

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.BFLOAT16
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        private val INFO = OperatorInfo("MatMul", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)

        private fun expandTensors(left: NDArrayTFJS, right: NDArrayTFJS): Pair<NDArrayTFJS, NDArrayTFJS> {
            return when {
                left.rank == right.rank -> left to right
                left.rank > right.rank -> {
                    val diff = left.rank - right.rank
                    val rightShape = Array(left.rank) { idx ->
                        if (idx < diff) 1 else right.shape[idx - diff]
                    }
                    val rightReshaped = reshape(right, rightShape)
                    left to rightReshaped
                }
                else -> {
                    val diff = right.rank - left.rank
                    val leftShape = Array(right.rank) { idx ->
                        if (idx < diff) 1 else left.shape[idx - diff]
                    }
                    val leftReshaped = reshape(left, leftShape)
                    leftReshaped to right
                }
            }
        }
    }


    override fun apply(context: Context, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val left = inputs[0]!!.data
            val right = inputs[1]!!.data
            val (leftActual, rightActual) = expandTensors(left, right)

            val output = leftActual.matMul(rightActual)
            return@tidy arrayOf(output)
        }
        return listOf(outputs[0].asTensor("Y"))
    }
}
