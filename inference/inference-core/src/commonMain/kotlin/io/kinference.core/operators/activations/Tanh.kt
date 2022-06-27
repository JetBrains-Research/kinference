package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.operator.*
import io.kinference.core.operators.math.tanh
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import kotlin.time.ExperimentalTime

sealed class Tanh(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(name, info, attributes, inputs, outputs) {
    companion object {
        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = tanh(value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = tanh(value)
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in TanhVer6.VERSION.asRange() -> TanhVer6(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Tanh operator: $version")
        }
    }
}

@ExperimentalTime
class TanhVer6(name: String, attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Tanh(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Tanh", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun activate(input: NDArray, contexts: Contexts<KIONNXData<*>>): NDArray = when (val type = input.type) {
        DataType.FLOAT -> input.map(Tanh.activateFloat)
        DataType.DOUBLE -> input.map(Tanh.activateDouble)
        else -> error("Unsupported data type for this operation: $type")
    }
}
