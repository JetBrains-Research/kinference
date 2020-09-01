package io.kinference.operators.activations

import io.kinference.ndarray.NDArray
import io.kinference.primitives.types.DataType
import io.kinference.attributes.Attribute
import io.kinference.ndarray.*
import io.kinference.operators.IOInfo
import io.kinference.operators.OperatorInfo
import io.kinference.operators.math.tanh
import kotlin.math.exp
import kotlin.math.max

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

@ExperimentalUnsignedTypes
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

@ExperimentalUnsignedTypes
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

@ExperimentalUnsignedTypes
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

@ExperimentalUnsignedTypes
class Erf(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Erf", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )
    }

    override fun activate(input: NDArray): NDArray = (input as NumberNDArray).erf()
}
