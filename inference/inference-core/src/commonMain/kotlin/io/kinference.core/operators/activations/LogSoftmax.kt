package io.kinference.core.operators.activations

import io.kinference.core.attributes.Attribute
import io.kinference.ndarray.arrays.NDArray
import io.kinference.ndarray.extensions.indexAxis
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import io.kinference.protobuf.message.AttributeProto
import kotlin.time.ExperimentalTime

sealed class LogSoftmax(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(info, attributes, inputs, outputs) {
    companion object {
        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in LogSoftmaxVer1.VERSION.asRange() -> LogSoftmaxVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of LogSoftmax operator: $version")
        }
    }
}

@OptIn(ExperimentalTime::class)
class LogSoftmaxVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : LogSoftmax(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val ATTRIBUTES_INFO = listOf(AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 1))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("LogSoftmax", ATTRIBUTES_INFO, INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    val axis: Int by attribute { it: Number -> it.toInt() }

    override fun activate(input: NDArray): NDArray {
        val actualAxis = input.indexAxis(axis)

        val output = Softmax.softmax(input, actualAxis)
        output.mapMutable(create("Log", output.type))
        return output
    }
}
