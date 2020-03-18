package org.jetbrains.research.kotlin.mpp.inference.deserializer.json

import kotlinx.serialization.Serializable

@Serializable
@Suppress("EnumEntryName")
enum class ActivationType {
    relu,
    sigmoid,
    tanh
}
