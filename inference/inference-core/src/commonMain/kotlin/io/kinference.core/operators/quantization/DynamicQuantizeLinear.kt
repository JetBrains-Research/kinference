package io.kinference.core.operators.quantization

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableUByteNDArray
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.createScalarNDArray
import io.kinference.core.operators.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import kotlin.math.*
import kotlin.time.ExperimentalTime

@ExperimentalTime
class DynamicQuantizeLinear(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.FLOAT), "x", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.UINT8), "y", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.FLOAT), "y_scale", optional = false),
            IOInfo(2, setOf(TensorProto.DataType.UINT8), "y_zero_point", optional = false)
        )

        private val INFO = OperatorInfo("DynamicQuantizeLinear", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private fun clip(x: Float, min: Float, max: Float) = when {
        x < min -> min
        x > max -> max
        else -> x
    }


    private fun Float.toUByte() = this.toUInt().toUByte()


    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs.first()!!.data as FloatNDArray

        val inputMin = min(0f, input.min())
        val inputMax = max(0f, input.max())

        val outputScale = (inputMax - inputMin) / 255f
        val outputScaleScalar = createScalarNDArray(DataType.FLOAT, outputScale)

        val outputZeroPoint = clip(round((-inputMin) / outputScale), 0f, 255f)
        val outputZeroPointScalar = createScalarNDArray(DataType.UBYTE, outputZeroPoint.toUByte())

        val output = allocateNDArray(DataType.UBYTE, input.strides) as MutableUByteNDArray

        for (i in 0 until input.linearSize) {
            output.array[i] = clip((round(input.array[i] / outputScale) + outputZeroPoint), 0f, 255f).toUByte()
        }

        return listOf(
            output.asTensor("y"),
            outputScaleScalar.asTensor("y_scale"),
            outputZeroPointScalar.asTensor("y_zero_point")
        )
    }
}
