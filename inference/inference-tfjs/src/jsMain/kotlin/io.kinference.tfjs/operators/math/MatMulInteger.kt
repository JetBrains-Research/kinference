package io.kinference.tfjs.operators.math

import io.kinference.protobuf.message.TensorProto
import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.operator.*
import io.kinference.profiler.ProfilingContext
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.extensions.*

sealed class MatMulInteger(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<TFJSTensor, TFJSTensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in MatMulIntegerVer10.VERSION.asRange() -> MatMulIntegerVer10(attributes, inputs, outputs)
            else -> error("Unsupported version of MatMulInteger operator: $version")
        }
    }
}

class MatMulIntegerVer10(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : MatMulInteger(INFO, attributes, inputs, outputs) {
    companion object {
        private val IN_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.INT8
        )

        private val OUT_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.INT32)

        private val INPUTS_INFO = listOf(
            IOInfo(0, IN_TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, IN_TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(2, IN_TYPE_CONSTRAINTS, "a_zero_point", optional = true),
            IOInfo(3, IN_TYPE_CONSTRAINTS, "b_zero_point", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, OUT_TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("MatMulInteger", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<TFJSTensor?>, profilingContext: ProfilingContext?, checkCancelled: () -> Unit): List<TFJSTensor?> {
        val outputs = tidy {
            val left = inputs[0]!!.data
            val right = inputs[1]!!.data
            val leftZP = inputs.getOrNull(2)?.data
            val rightZP = inputs.getOrNull(3)?.data

            val leftWithZp = if (leftZP != null) left - leftZP else left
            val rightWithZp = if (rightZP != null) right - rightZP else right

            val (leftExpanded, rightExpanded) = MatMul.expandTensors(leftWithZp, rightWithZp)

            return@tidy arrayOf(leftExpanded.matMul(rightExpanded))
        }

        return listOf(outputs[0].asTensor("Y"))
    }
}

