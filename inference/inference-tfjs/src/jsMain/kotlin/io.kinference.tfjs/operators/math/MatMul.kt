package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.matMul
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class MatMul(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        internal fun expandTensors(left: NumberNDArrayTFJS, right: NumberNDArrayTFJS): Pair<NumberNDArrayTFJS, NumberNDArrayTFJS> {
            return when {
                left.rank == right.rank -> left to right
                left.rank > right.rank -> {
                    val diff = left.rank - right.rank
                    val rightShape = IntArray(left.rank) { idx ->
                        if (idx < diff) 1 else right.shape[idx - diff]
                    }
                    val rightReshaped = right.reshape(rightShape)
                    left to rightReshaped
                }

                else -> {
                    val diff = right.rank - left.rank
                    val leftShape = IntArray(right.rank) { idx ->
                        if (idx < diff) 1 else left.shape[idx - diff]
                    }
                    val leftReshaped = left.reshape(leftShape)
                    leftReshaped to right
                }
            }
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in MatMulVer1.VERSION.asRange() -> MatMulVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of MatMul operator: $version")
            }
    }
}

class MatMulVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    MatMul(name, INFO, attributes, inputs, outputs) {
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


    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val output = tidyNDArray {
            val left = inputs[0]!!.data as NumberNDArrayTFJS
            val right = inputs[1]!!.data as NumberNDArrayTFJS
            val (leftActual, rightActual) = expandTensors(left, right)

            return@tidyNDArray leftActual.matmul(rightActual)
        }
        return listOf(output.asTensor("Y"))
    }
}
