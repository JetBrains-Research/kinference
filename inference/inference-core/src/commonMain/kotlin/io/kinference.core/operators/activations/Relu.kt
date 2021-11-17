package io.kinference.core.operators.activations

import io.kinference.core.attributes.Attribute
import io.kinference.core.operators.*
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import kotlin.math.max
import kotlin.time.ExperimentalTime

sealed class Relu(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(info, attributes, inputs, outputs) {
    companion object {
        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = max(0.0f, value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = max(0.0, value)
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6, untilVersion = 14)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ReluVer6.VERSION.asRange() -> ReluVer6(attributes, inputs, outputs)
            else -> error("Unsupported version of Relu operator: $version")
        }
    }
}

@ExperimentalTime
class ReluVer6(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Relu(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6, untilVersion = 14)
        private val INFO = OperatorInfo("Relu", emptyMap(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun activate(input: NDArray): NDArray = when (val type = input.type) {
        DataType.FLOAT -> input.map(Relu.activateFloat)
        DataType.DOUBLE -> input.map(Relu.activateDouble)
        else -> error("Unsupported data type for this operation: $type")
    }
}
