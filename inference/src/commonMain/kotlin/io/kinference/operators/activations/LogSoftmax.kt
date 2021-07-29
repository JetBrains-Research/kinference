package io.kinference.operators.activations

import io.kinference.attributes.Attribute
import io.kinference.ndarray.arrays.NDArray
import io.kinference.ndarray.extensions.indexAxis
import io.kinference.operators.*
import io.kinference.protobuf.message.AttributeProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class LogSoftmax(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 1)
        )

        private val INFO = OperatorInfo("LogSoftmax", ATTRIBUTES_INFO,
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false, differentiable = true)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false, differentiable = true))
        )

    }

    val axis: Int by attribute() { it: Number -> it.toInt() }

    override fun activate(input: NDArray): NDArray {
        val actualAxis = input.indexAxis(axis)

        val output = Softmax.softmax(input, actualAxis)
        output.mapMutable(create("Log", output.type))
        return output
    }
}
