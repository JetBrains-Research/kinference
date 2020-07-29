package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.operators.*
import kotlin.math.exp
import kotlin.math.max

class Identity(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INFO = OperatorInfo("Identity", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = input
}

class Relu(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Relu", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )

        inline fun activate(value: Any): Any {
            return when (value) {
                is Float -> max(0.0f, value)
                is Double -> max(0.0, value)
                else -> error("Unsupported operation")
            }
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = input.mapElements(Companion::activate)
}

class Sigmoid(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Sigmoid", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )

        inline fun activate(value: Any): Any {
            return when (value) {
                is Float -> (1.0 / (1.0 + exp(-value))).toFloat()
                is Double -> 1.0 / (1.0 + exp(-value))
                else -> error("Unsupported operation")
            }
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = input.mapElements(Companion::activate)
}

class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Tanh", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )

        inline fun activate(value: Any): Any {
            return when (value) {
                is Float -> ((exp(2.0 * value) - 1.0) / (exp(2.0 * value) + 1.0)).toFloat()
                is Double -> (exp(2.0 * value) - 1.0) / (exp(2.0 * value) + 1.0)
                else -> error("Unsupported operation")
            }
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = input.mapElements(Companion::activate)
}
