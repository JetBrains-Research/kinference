package io.kinference.core.operators.math

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.operators.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class MatMulIntegerToFloat(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val IN_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.INT8
        )

        private val OUT_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.FLOAT)

        private val INPUTS_INFO = listOf(
            IOInfo(0, IN_TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, IN_TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(2, OUT_TYPE_CONSTRAINTS, "a_scale", optional = false),
            IOInfo(3, OUT_TYPE_CONSTRAINTS, "b_scale", optional = false),
            IOInfo(4, IN_TYPE_CONSTRAINTS, "a_zero_point", optional = true),
            IOInfo(5, IN_TYPE_CONSTRAINTS, "b_zero_point", optional = true),
            IOInfo(6, OUT_TYPE_CONSTRAINTS, "bias", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, OUT_TYPE_CONSTRAINTS, "Y", optional = false))

        private val INFO = OperatorInfo("MatMulIntegerToFloat", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val left = inputs[0]!!.data as NumberNDArray
        val right = inputs[1]!!.data as NumberNDArray
        val leftScale = inputs[2]!!.data as FloatNDArray
        val rightScale = inputs[3]!!.data as FloatNDArray

        val leftZeroPoint = inputs[4]?.data as? NumberNDArray
        val rightZeroPoint = inputs[5]?.data as? NumberNDArray

        val bias = inputs[6]?.data as? FloatNDArray

        val leftDequant = left.dequantize(leftZeroPoint, leftScale) as NumberNDArray
        val rightDequant = right.dequantize(rightZeroPoint, rightScale) as NumberNDArray

        val outputArray = leftDequant.matmul(rightDequant)

        if (bias != null) {
            outputArray.plusAssign(bias)
        }

        return listOf(outputArray.asTensor("Y"))
    }
}

