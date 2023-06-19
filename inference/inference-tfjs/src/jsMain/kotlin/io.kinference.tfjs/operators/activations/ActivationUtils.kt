package io.kinference.tfjs.operators.activations

import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.*

internal fun NumberNDArrayTFJS.activate(name: String) = when (name) {
    "Sigmoid" -> this.sigmoid()
    "Tanh" -> this.tanh()
    "Relu" -> this.relu()
    "Log" -> this.log()
    else -> error("Unsupported activation function: $name")
}
