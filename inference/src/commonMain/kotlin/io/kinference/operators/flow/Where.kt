package io.kinference.operators.flow

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.acceptTriple
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.operators.*
import io.kinference.primitives.types.DataType
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.TensorProto

@ExperimentalTime
class Where(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.BOOL), "condition", optional = false),
            IOInfo(1, ALL_DATA_TYPES, "X", optional = false),
            IOInfo(2, ALL_DATA_TYPES, "Y", optional = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val INFO = OperatorInfo("Where", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val condition = inputs[0]!!.data
        val left = inputs[1]!!.data
        val right = inputs[2]!!.data

        require(left.type == right.type)

        val outputArray = Broadcasting.applyWithBroadcast(listOf(condition, left, right), left.type) { inputs, dest ->
            val (condition, left, right) = inputs

            val conditionPoint = (condition as BooleanNDArray).array.pointer()

            when (left.type) {
                DataType.FLOAT -> {
                    val leftPoint = (left as FloatNDArray).array.pointer()
                    val rightPoint = (right as FloatNDArray).array.pointer()
                    val destPoint = (dest as FloatNDArray).array.pointer()

                    destPoint.acceptTriple(leftPoint, rightPoint, conditionPoint, dest.linearSize) { _, left, right, cond -> if (cond) left else right }
                }
                DataType.BYTE -> {
                    val leftPoint = (left as ByteNDArray).array.pointer()
                    val rightPoint = (right as ByteNDArray).array.pointer()
                    val destPoint = (dest as ByteNDArray).array.pointer()

                    destPoint.acceptTriple(leftPoint, rightPoint, conditionPoint, dest.linearSize) { _, left, right, cond -> if (cond) left else right }
                }
                DataType.SHORT -> {
                    val leftPoint = (left as ShortNDArray).array.pointer()
                    val rightPoint = (right as ShortNDArray).array.pointer()
                    val destPoint = (dest as ShortNDArray).array.pointer()

                    destPoint.acceptTriple(leftPoint, rightPoint, conditionPoint, dest.linearSize) { _, left, right, cond -> if (cond) left else right }
                }
                DataType.INT -> {
                    val leftPoint = (left as IntNDArray).array.pointer()
                    val rightPoint = (right as IntNDArray).array.pointer()
                    val destPoint = (dest as IntNDArray).array.pointer()

                    destPoint.acceptTriple(leftPoint, rightPoint, conditionPoint, dest.linearSize) { _, left, right, cond -> if (cond) left else right }
                }
                DataType.LONG -> {
                    val leftPoint = (left as LongNDArray).array.pointer()
                    val rightPoint = (right as LongNDArray).array.pointer()
                    val destPoint = (dest as LongNDArray).array.pointer()

                    destPoint.acceptTriple(leftPoint, rightPoint, conditionPoint, dest.linearSize) { _, left, right, cond -> if (cond) left else right }
                }
                DataType.UBYTE -> {
                    val leftPoint = (left as UByteNDArray).array.pointer()
                    val rightPoint = (right as UByteNDArray).array.pointer()
                    val destPoint = (dest as UByteNDArray).array.pointer()

                    destPoint.acceptTriple(leftPoint, rightPoint, conditionPoint, dest.linearSize) { _, left, right, cond -> if (cond) left else right }
                }
                DataType.ULONG -> {
                    val leftPoint = (left as ULongNDArray).array.pointer()
                    val rightPoint = (right as ULongNDArray).array.pointer()
                    val destPoint = (dest as ULongNDArray).array.pointer()

                    destPoint.acceptTriple(leftPoint, rightPoint, conditionPoint, dest.linearSize) { _, left, right, cond -> if (cond) left else right }
                }
                DataType.DOUBLE -> {
                    val leftPoint = (left as DoubleNDArray).array.pointer()
                    val rightPoint = (right as DoubleNDArray).array.pointer()
                    val destPoint = (dest as DoubleNDArray).array.pointer()

                    destPoint.acceptTriple(leftPoint, rightPoint, conditionPoint, dest.linearSize) { _, left, right, cond -> if (cond) left else right }
                }
                DataType.BOOLEAN -> {
                    val leftPoint = (left as BooleanNDArray).array.pointer()
                    val rightPoint = (right as BooleanNDArray).array.pointer()
                    val destPoint = (dest as BooleanNDArray).array.pointer()

                    destPoint.acceptTriple(leftPoint, rightPoint, conditionPoint, dest.linearSize) { _, left, right, cond -> if (cond) left else right }
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        return listOf(outputArray.asTensor())
    }
}
