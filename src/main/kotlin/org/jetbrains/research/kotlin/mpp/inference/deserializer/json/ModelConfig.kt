package org.jetbrains.research.kotlin.mpp.inference.deserializer.json

import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

@Serializable
sealed class ModelConfig {
    abstract val name: String
    abstract val layers: List<LayerScheme>

    @Serializable
    data class Sequential(
        override val name: String,
        override val layers: List<LayerScheme>
    ) : ModelConfig() {
        @Transient
        val batchInputShape = layers.first().config.batch_input_shape!!.filterNotNull()
    }
}
