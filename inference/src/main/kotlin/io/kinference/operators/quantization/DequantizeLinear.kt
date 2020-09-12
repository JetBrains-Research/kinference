package io.kinference.operators.quantization

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.*
import io.kinference.ndarray.extensions.isScalar
import io.kinference.onnx.AttributeProto
import io.kinference.onnx.TensorProto
import io.kinference.operators.*

class DequantizeLinear(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val IN_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.INT8,
            TensorProto.DataType.UINT8
        )

        private val OUT_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), required = false, default = 1)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, IN_TYPE_CONSTRAINTS, "x", optional = false),
            IOInfo(0, OUT_TYPE_CONSTRAINTS, "x_scale", optional = false),
            IOInfo(0, IN_TYPE_CONSTRAINTS, "x_zero_point", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, OUT_TYPE_CONSTRAINTS, "y", optional = false))

        private val INFO = OperatorInfo("DequantizeLinear", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        private fun NDArray.canQuantizePerAxis(axis: Int, zeroPoint: NDArray?, scale: NDArray): Boolean {
            return scale.rank == 1 && scale.linearSize == shape[axis] && (zeroPoint == null || zeroPoint.rank == 1 && zeroPoint.linearSize == shape[axis])
        }

        private fun canQuantizePerTensor(zeroPoint: NDArray?, scale: NDArray): Boolean {
            return scale.isScalar() && (zeroPoint == null || zeroPoint.isScalar())
        }

        @ExperimentalUnsignedTypes
        private fun NDArray.quantize(zeroPoint: NDArray?, scale: NDArray, axis: Int): NDArray {
            this as NumberNDArray; scale as FloatNDArray
            val output = MutableFloatNDArray(FloatArray(this.linearSize), this.strides)
            when {
                canQuantizePerAxis(axis, zeroPoint, scale) -> {
                    val axisBlockSize = shape[axis]
                    for (i in 0 until output.linearSize) {
                        val idx = i % axisBlockSize
                        val zero = (zeroPoint?.get(idx) as? Number)?.toFloat() ?: 0f
                        output[i] = ((this[i] as Number).toFloat() - zero) * scale[idx]
                    }
                }
                canQuantizePerTensor(zeroPoint, scale) -> {
                    val zero = (zeroPoint?.get(0) as? Number)?.toFloat() ?: 0f
                    for (i in 0 until output.linearSize) {
                        output[i] = ((this[i] as Number).toFloat() - zero) * scale[0]
                    }
                }
                else -> error("Cannot perform dequantization. Scale and zero point tensors should be either scalars or 1D tensors containing ${shape[axis]} elements")
            }
            return output
        }
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    @ExperimentalUnsignedTypes
    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val input = inputs[0]!!.data
        val scale = inputs[1]!!.data
        val zeroPoint = inputs.getOrNull(2)?.data

        require(zeroPoint == null || scale.shape.contentEquals(zeroPoint.shape)) { "Zero point and scale tensors should have the same dims" }

        return listOf(input.quantize(zeroPoint, scale, axis).asTensor("y"))
    }
}
