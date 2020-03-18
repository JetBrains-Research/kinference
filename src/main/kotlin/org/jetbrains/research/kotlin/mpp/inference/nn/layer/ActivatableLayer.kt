package org.jetbrains.research.kotlin.mpp.inference.nn.layer

import org.jetbrains.research.kotlin.mpp.inference.nn.activation.ActivationFunction
import org.jetbrains.research.kotlin.mpp.inference.nn.parameters.Parameters
import scientifik.kmath.structures.NDStructure

abstract class ActivatableLayer<T : NDStructure<*>>(name: String, params: Parameters) : Layer<T>(name, params) {
    protected abstract val activationFunction: ActivationFunction

    abstract fun activate()
}
