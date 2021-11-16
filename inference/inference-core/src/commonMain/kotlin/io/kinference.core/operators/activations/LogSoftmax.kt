package io.kinference.core.operators.activations

import io.kinference.core.attributes.Attribute
import io.kinference.ndarray.arrays.NDArray
import io.kinference.ndarray.extensions.indexAxis
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import io.kinference.protobuf.message.AttributeProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class LogSoftmax(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val ATTRIBUTES_INFO = listOf(AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 1))

        private val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("LogSoftmax", ATTRIBUTES_INFO, INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> LogSoftmax(attributes, inputs, outputs)
            else -> error("Unsupported version of LogSoftmax operator: $version")
        }
    }

    val axis: Int by attribute { it: Number -> it.toInt() }

    override fun activate(input: NDArray): NDArray {
        val actualAxis = input.indexAxis(axis)

        val output = Softmax.softmax(input, actualAxis)
        output.mapMutable(create("Log", output.type))
        return output
    }
}
