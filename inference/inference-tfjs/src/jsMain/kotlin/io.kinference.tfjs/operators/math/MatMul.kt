package io.kinference.tfjs.operators.math

import io.kinference.protobuf.message.TensorProto
import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.operator.*
import io.kinference.profiler.ProfilingContext
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.core.NDArrayTFJS
import io.kinference.tfjs.externals.core.reshape
import io.kinference.tfjs.externals.extensions.matMul
import io.kinference.tfjs.externals.extensions.tidy

sealed class MatMul(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<TFJSTensor, TFJSTensor>(info, attributes, inputs, outputs) {
    companion object {
        internal fun expandTensors(left: NDArrayTFJS, right: NDArrayTFJS): Pair<NDArrayTFJS, NDArrayTFJS> {
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

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in MatMulVer1.VERSION.asRange() -> MatMulVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of MatMul operator: $version")
        }
    }
}

class MatMulVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : MatMul(INFO, attributes, inputs, outputs) {
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

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("MatMul", emptySet(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }


    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<TFJSTensor?>, profilingContext: ProfilingContext?, checkCancelled: () -> Unit): List<TFJSTensor?> {
        val outputs = tidy {
            val left = inputs[0]!!.data
            val right = inputs[1]!!.data
            val (leftActual, rightActual) = MatMul.expandTensors(left, right)

            val output = leftActual.matMul(rightActual)
            return@tidy arrayOf(output)
        }
        return listOf(outputs[0].asTensor("Y"))
    }
}
