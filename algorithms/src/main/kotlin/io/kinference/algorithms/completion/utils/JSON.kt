package io.kinference.algorithms.completion.utils

import kotlinx.serialization.DeserializationStrategy
import kotlinx.serialization.SerializationStrategy
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonConfiguration

internal object JSON {
    val json = Json(JsonConfiguration.Stable)

    inline fun <reified T : Any> string(serializer: SerializationStrategy<T>, value: T): String {
        return json.stringify(serializer, value)
    }

    inline fun <reified T> parse(serializer: DeserializationStrategy<T>, value: String): T {
        return json.parse(serializer, value)
    }
}
