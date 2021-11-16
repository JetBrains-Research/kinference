package io.kinference.core.operators.activations

import io.kinference.core.attributes.Attribute
import io.kinference.core.operators.*
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import kotlin.math.ln
import kotlin.time.ExperimentalTime

sealed class Log(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in LogVer6.VERSION.asRange() -> LogVer6(attributes, inputs, outputs)
            else -> error("Unsupported version of Log operator: $version")
        }

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = ln(value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = ln(value)
        }
    }
}

@ExperimentalTime
class LogVer6(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Log(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Log", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun activate(input: NDArray): NDArray = when (val type = input.type) {
        DataType.FLOAT -> input.map(Log.activateFloat)
        DataType.DOUBLE -> input.map(Log.activateDouble)
        else -> error("Unsupported data type for this operation: $type")
    }
}
