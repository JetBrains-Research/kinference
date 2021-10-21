package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.extensions.indexAxis
import io.kinference.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class Flatten(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input", optional = false, differentiable = true),
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false, differentiable = true))

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), default = 1, required = false)
        )

        private val INFO = OperatorInfo("Flatten", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    val axis: Int by attribute() { it: Number -> it.toInt() }

    private fun makeShape(shape: IntArray, axis: Int): IntArray {
        val firstDimension = shape.slice(0 until axis).fold(1, Int::times)
        val secondDimension = shape.slice(axis until shape.size).fold(1, Int::times)

        return intArrayOf(firstDimension, secondDimension)
    }

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val input = inputs[0]!!.data
        val actualAxis = input.indexAxis(axis)

        val newShape = makeShape(input.shape, actualAxis)
        return listOf(input.toMutable().reshape(newShape).asTensor("output"))
    }

}