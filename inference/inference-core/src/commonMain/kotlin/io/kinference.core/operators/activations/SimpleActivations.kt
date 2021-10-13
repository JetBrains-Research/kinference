package io.kinference.core.operators.activations

import io.kinference.core.attributes.Attribute
import io.kinference.ndarray.arrays.*
import io.kinference.core.operators.*
import io.kinference.core.operators.math.tanh
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import kotlin.math.*
import kotlin.time.ExperimentalTime

@ExperimentalTime
class Identity(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INFO = OperatorInfo("Identity", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )
    }

    override fun activate(input: NDArray): NDArray = input
}

@ExperimentalTime
class Relu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Relu", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = max(0.0f, value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = max(0.0, value)
        }
    }

    override fun activate(input: NDArray): NDArray = when (input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported operation")
    }
}

@ExperimentalTime
class LeakyRelu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("LeakyRelu",
            listOf(AttributeInfo("alpha", setOf(AttributeProto.AttributeType.FLOAT), default = 0.01f)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )
    }

    val alpha: Float by attribute()

    private val activateFloat: FloatMap = object : FloatMap {
        override fun apply(value: Float): Float = if (value < 0) value * alpha else value
    }

    private val activateDouble: DoubleMap = object : DoubleMap {
        override fun apply(value: Double): Double = if (value < 0) value * alpha else value
    }

    override fun activate(input: NDArray): NDArray = when (input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported operation")
    }
}

@ExperimentalTime
class Sigmoid(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Sigmoid", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = 1.0f / (1.0f + exp(-value))
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = 1.0 / (1.0 + exp(-value))
        }
    }

    override fun activate(input: NDArray): NDArray = when (input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported operation")
    }
}

@ExperimentalTime
class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Tanh", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = tanh(value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = tanh(value)
        }
    }

    override fun activate(input: NDArray): NDArray = when (input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported operation")
    }
}

@ExperimentalTime
class Erf(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Erf", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )
    }

    override fun activate(input: NDArray): NDArray = (input.toMutable() as MutableNumberNDArray).erf()
}

@ExperimentalTime
class Log(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Log", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false, differentiable = true)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false, differentiable = true))
        )

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = ln(value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = ln(value)
        }
    }

    override fun activate(input: NDArray): NDArray = when (input.type) {
        DataType.FLOAT -> input.map(activateFloat)
        DataType.DOUBLE -> input.map(activateDouble)
        else -> error("Unsupported operation")
    }
}
