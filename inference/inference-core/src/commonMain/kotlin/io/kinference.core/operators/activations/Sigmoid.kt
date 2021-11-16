package io.kinference.core.operators.activations

import io.kinference.core.attributes.Attribute
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import kotlin.math.exp
import kotlin.time.ExperimentalTime

sealed class Sigmoid(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(info, attributes, inputs, outputs) {
    companion object {
        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = 1.0f / (1.0f + exp(-value))
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = 1.0 / (1.0 + exp(-value))
        }

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in SigmoidVer6.VERSION.asRange() -> SigmoidVer6(attributes, inputs, outputs)
            else -> error("Unsupported version of Sigmoid operator: $version")
        }
    }
}

@ExperimentalTime
class SigmoidVer6(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Sigmoid(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Sigmoid", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun activate(input: NDArray): NDArray = when (val type = input.type) {
        DataType.FLOAT -> input.map(Sigmoid.activateFloat)
        DataType.DOUBLE -> input.map(Sigmoid.activateDouble)
        else -> error("Unsupported data type for this operation: $type")
    }
}
