package io.kinference.operators.quantization

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.FloatNDArray
import io.kinference.ndarray.MutableUByteNDArray
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.createScalarNDArray
import io.kinference.onnx.TensorProto
import io.kinference.operators.AttributeInfo
import io.kinference.operators.IOInfo
import io.kinference.operators.Operator
import io.kinference.operators.OperatorInfo
import io.kinference.primitives.types.DataType
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

class DynamicQuantizeLinear(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
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

    @ExperimentalUnsignedTypes
    private fun Float.toUByte() = this.toUInt().toUByte()

    @ExperimentalUnsignedTypes
    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
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
