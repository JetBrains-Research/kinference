package io.kinference.operators.math

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.ndarray.arrays.pointers.mapTo
import io.kinference.ndarray.extensions.matmul
import io.kinference.ndarray.arrays.tiled.IntTiledArray
import io.kinference.onnx.TensorProto
import io.kinference.operators.IOInfo
import io.kinference.operators.Operator
import io.kinference.operators.OperatorInfo

class MatMulInteger(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
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

        private val INFO = OperatorInfo("MatMulInteger", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)

        @ExperimentalUnsignedTypes
        private fun NumberNDArray.toIntNDArray(): IntNDArray {
            val result = IntNDArray(IntTiledArray(this.strides), strides)
            when (this) {
                is UByteNDArray -> {
                    this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toInt() }
                }
                is ByteNDArray -> {
                    this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toInt() }
                }
                else -> error("Unsupported data type: $type")
            }

            return result
        }
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val first = inputs[0]!!.data as NumberNDArray
        val second = inputs[1]!!.data as NumberNDArray
        val firstZero = inputs.getOrNull(2)?.data as? NumberNDArray
        val secondZero = inputs.getOrNull(3)?.data as? NumberNDArray

        val firstBiased = if (firstZero == null) first.toIntNDArray() else first.withZeroPoint(firstZero)
        val secondBiased = if (secondZero == null) second.toIntNDArray() else second.withZeroPoint(secondZero)

        return listOf((firstBiased matmul secondBiased).asTensor("y"))
    }
}
