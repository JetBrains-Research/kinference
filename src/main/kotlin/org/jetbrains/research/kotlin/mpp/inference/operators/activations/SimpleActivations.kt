package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import kotlin.math.exp

class Identity(attributes: Map<String, Attribute<Any>> = emptyMap()) : Activation("Identity", TYPE_CONSTRAINTS, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES
    }

    override fun activate(input: Tensor): Tensor = input
}

class Relu(attributes: Map<String, Attribute<Any>> = emptyMap()) : Activation("Relu", TYPE_CONSTRAINTS, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES
    }

    override fun activate(input: Tensor): Tensor = input.mapElements { x -> max(0, x as Number) }
}

class Sigmoid(attributes: Map<String, Attribute<Any>> = emptyMap()) : Activation("Sigmoid", TYPE_CONSTRAINTS, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES
    }

    override fun activate(input: Tensor): Tensor = input.mapElements { x ->
        1.0 / (1.0 + exp(-(x as Number).toDouble()))
    }
}

class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap()) : Activation("Tanh", TYPE_CONSTRAINTS, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES
    }

    override fun activate(input: Tensor): Tensor = input.mapElements { x ->
        x as Number
        (exp(2.0 * x.toDouble()) - 1.0) / (exp(2.0 * x.toDouble()) + 1.0)
    }
}
