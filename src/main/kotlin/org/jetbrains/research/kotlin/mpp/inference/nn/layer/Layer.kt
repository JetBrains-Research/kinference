package org.jetbrains.research.kotlin.mpp.inference.nn.layer

import org.jetbrains.research.kotlin.mpp.inference.deserializer.json.LayerConfig
import org.jetbrains.research.kotlin.mpp.inference.deserializer.json.LayerScheme
import org.jetbrains.research.kotlin.mpp.inference.nn.activation.ActivationFunction
import org.jetbrains.research.kotlin.mpp.inference.nn.layer.dense.DenseLayer
import org.jetbrains.research.kotlin.mpp.inference.nn.layer.dense.DenseParameters
import scientifik.kmath.structures.NDStructure

abstract class Layer<T : NDStructure<*>>(val name: String, val params: Parameters<T>) {
    data class Parameters<T : NDStructure<*>>(val weights: T?, val biases: T?)

    companion object {
        @Suppress("UNCHECKED_CAST")
        fun create(config: LayerScheme, params: Parameters<*>) = when (config.type) {
            LayerScheme.Type.DENSE -> DenseLayer(
                name = config.config.name,
                params = params as DenseParameters,
                activationFunction = ActivationFunction[(config.config as LayerConfig.Dense).activation]
            )
            LayerScheme.Type.DROPOUT -> null
        }
    }
}
