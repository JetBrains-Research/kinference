package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.operators.InputInfo
import org.jetbrains.research.kotlin.mpp.inference.operators.OperatorInfo
import org.jetbrains.research.kotlin.mpp.inference.operators.OutputInfo
import kotlin.math.exp

class Identity(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INFO = OperatorInfo("Identity", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )
    }

    override fun activate(input: Tensor): Tensor = input
}

class Relu(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Relu", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )
    }

    override fun activate(input: Tensor): Tensor = input.mapElements { x -> max(0, x as Number) }
}

class Sigmoid(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Sigmoid", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )
    }

    override fun activate(input: Tensor): Tensor = input.mapElements { x ->
        1.0 / (1.0 + exp(-(x as Number).toDouble()))
    }
}

class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap(), usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Tanh", emptyMap(),
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )
    }

    override fun activate(input: Tensor): Tensor = input.mapElements { x ->
        x as Number
        (exp(2.0 * x.toDouble()) - 1.0) / (exp(2.0 * x.toDouble()) + 1.0)
    }
}
