package org.jetbrains.research.kotlin.mpp.inference.deserializer.json

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonConfiguration

@Serializable
sealed class ModelScheme {
    abstract val config: ModelConfig

    @Serializable
    @SerialName("Sequential")
    @Suppress("UNUSED")
    data class SequentialScheme(override val config: ModelConfig.Sequential) : ModelScheme()

    companion object {
        private val json = Json(configuration = JsonConfiguration.Stable.copy(strictMode = false, classDiscriminator = "class_name"))

        fun parse(config: String) = json.parse(serializer(), config)
    }
}
