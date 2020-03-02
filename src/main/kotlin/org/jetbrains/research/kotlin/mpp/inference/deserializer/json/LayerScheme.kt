package org.jetbrains.research.kotlin.mpp.inference.deserializer.json

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

@Serializable
sealed class LayerScheme {
    abstract val type: Type
    abstract val config: LayerConfig

    @Serializable
    @SerialName("Dense")
    @Suppress("UNUSED")
    data class DenseLayerScheme(
        @Transient override val type: Type = Type.DENSE,
        override val config: LayerConfig.Dense
    ) : LayerScheme()

    @Serializable
    @SerialName("Dropout")
    @Suppress("UNUSED")
    data class DropoutLayerScheme(
        @Transient override val type: Type = Type.DROPOUT,
        override val config: LayerConfig.Dropout
    ) : LayerScheme()

    enum class Type {
        DENSE,
        DROPOUT
    }
}
