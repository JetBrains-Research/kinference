package io.kinference.core.operators.activations

import io.kinference.core.attributes.Attribute
import io.kinference.ndarray.arrays.*
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import io.kinference.core.operators.math.tanh
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import kotlin.math.*
import kotlin.time.ExperimentalTime

@ExperimentalTime
class Identity(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 14)
        private val INFO = OperatorInfo("Identity", emptyMap(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> Identity(attributes, inputs, outputs)
            else -> error("Unsupported version of Identity operator: $version")
        }
    }

    override fun activate(input: NDArray): NDArray = allocateNDArray(input.type, input.shape).apply { copyFrom(0, input) }
}

@ExperimentalTime
class Relu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        private val VERSION = VersionInfo(sinceVersion = 6, untilVersion = 14)
        private val INFO = OperatorInfo("Relu", emptyMap(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> Relu(attributes, inputs, outputs)
            else -> error("Unsupported version of Relu operator: $version")
        }

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = max(0.0f, value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = max(0.0, value)
        }
    }

    override fun activate(input: NDArray): NDArray = when (val type = input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported data type for this operation: $type")
    }
}

@ExperimentalTime
class LeakyRelu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTE_INFO = listOf(AttributeInfo("alpha", setOf(AttributeProto.AttributeType.FLOAT), default = 0.01f))

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        private val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("LeakyRelu", ATTRIBUTE_INFO, INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> LeakyRelu(attributes, inputs, outputs)
            else -> error("Unsupported version of LeakyRelu operator: $version")
        }
    }

    val alpha: Float by attribute()

    private val activateFloat: FloatMap = object : FloatMap {
        override fun apply(value: Float): Float = if (value < 0) value * alpha else value
    }

    private val activateDouble: DoubleMap = object : DoubleMap {
        override fun apply(value: Double): Double = if (value < 0) value * alpha else value
    }

    override fun activate(input: NDArray): NDArray = when (val type = input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported data type for this operation: $type")
    }
}

@ExperimentalTime
class Sigmoid(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        private val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Sigmoid", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> Sigmoid(attributes, inputs, outputs)
            else -> error("Unsupported version of Sigmoid operator: $version")
        }

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = 1.0f / (1.0f + exp(-value))
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = 1.0 / (1.0 + exp(-value))
        }
    }

    override fun activate(input: NDArray): NDArray = when (val type = input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported data type for this operation: $type")
    }
}

@ExperimentalTime
class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Tanh", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> Tanh(attributes, inputs, outputs)
            else -> error("Unsupported version of Tanh operator: $version")
        }

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = tanh(value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = tanh(value)
        }
    }

    override fun activate(input: NDArray): NDArray = when (val type = input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported data type for this operation: $type")
    }
}

@ExperimentalTime
class Erf(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("Erf", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> Erf(attributes, inputs, outputs)
            else -> error("Unsupported version of Erf operator: $version")
        }
    }

    override fun activate(input: NDArray): NDArray = (input.toMutable() as MutableNumberNDArray).erf()
}

@ExperimentalTime
class Log(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        private val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Log", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> Log(attributes, inputs, outputs)
            else -> error("Unsupported version of Log operator: $version")
        }

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = ln(value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = ln(value)
        }
    }

    override fun activate(input: NDArray): NDArray = when (val type = input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported data type for this operation: $type")
    }
}
