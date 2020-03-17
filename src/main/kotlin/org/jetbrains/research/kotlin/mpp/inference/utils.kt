package org.jetbrains.research.kotlin.mpp.inference

import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.cbor.Cbor
import java.io.File

@Serializable
data class Mapping(val data: Map<String, FloatArray>) {
    operator fun get(fieldName: String): FloatArray? = data[fieldName]
}
fun loadMapping(name: String): Mapping = org.jetbrains.research.kotlin.mpp.inference.Cbor.parse(File(name).readBytes(), Mapping.serializer())


object Cbor {
    val cbor = Cbor()

    @Suppress("unused")
    inline fun <reified T> bytes(value: T, serializer: KSerializer<T>) = cbor.dump(serializer, value)

    inline fun <reified T> parse(value: ByteArray, serializer: KSerializer<T>) = cbor.load(serializer, value)
}
