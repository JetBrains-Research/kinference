package io.kinference.operators.quantization

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.ndarray.extensions.quantizeMatMul
import io.kinference.operators.*
import io.kinference.operators.quantization.DynamicQuantizeLinear.Companion.dynamicQuantize
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class DynamicQuantizeMatMul(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val BYTES_TYPES = setOf(
            TensorProto.DataType.INT8,
            TensorProto.DataType.UINT8
        )

        private val FLOAT_TYPE = setOf(TensorProto.DataType.FLOAT)

        private val INPUTS_INFO = listOf(
            IOInfo(0, FLOAT_TYPE, "A", optional = false),
            IOInfo(1, BYTES_TYPES, "B", optional = false),
            IOInfo(2, FLOAT_TYPE, "b_scale", optional = false),
            IOInfo(3, BYTES_TYPES, "b_zero_point", optional = true),
            IOInfo(4, FLOAT_TYPE, "bias", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, FLOAT_TYPE, "Y", optional = false))

        private val INFO = OperatorInfo("DynamicQuantizeMatMul", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val left = inputs[0]!!.data as FloatNDArray
        val quantizedRight = inputs[1]!!.data as NumberNDArray

        val rightScale = inputs[2]!!.data as FloatNDArray
        val rightZeroPoint = inputs[3]?.data as? NumberNDArray
        val bias = inputs[4]?.data as? FloatNDArray

        val (quantizedLeft, leftScale, leftZeroPoint) = left.dynamicQuantize()
        val output = quantizeMatMul(quantizedLeft, quantizedRight, leftZeroPoint, rightZeroPoint, leftScale, rightScale)

        if (bias != null) {
            output.plusAssign(bias)
        }

        return listOf(output.asTensor("Y"))
    }
}
