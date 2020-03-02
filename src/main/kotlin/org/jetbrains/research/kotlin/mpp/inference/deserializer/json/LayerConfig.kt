package org.jetbrains.research.kotlin.mpp.inference.deserializer.json

import kotlinx.serialization.Serializable

@Serializable
@Suppress("PropertyName")
sealed class LayerConfig {
    abstract val name: String
    abstract val trainable: Boolean
    abstract val batch_input_shape: List<Int?>?
    abstract val dtype: String?

    @Serializable
    data class Dense(
        override val name: String,
        override val trainable: Boolean,
        override val batch_input_shape: List<Int?>? = null,
        override val dtype: String? = null,
        val units: Int,
        val activation: ActivationType,
        val use_bias: Boolean
    ) : LayerConfig()

    @Serializable
    data class Dropout(
        override val name: String,
        override val trainable: Boolean,
        override val batch_input_shape: List<Int?>? = null,
        override val dtype: String? = null,
        val rate: Double
    ) : LayerConfig()
}
