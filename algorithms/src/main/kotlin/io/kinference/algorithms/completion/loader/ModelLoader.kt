package io.kinference.algorithms.completion.loader

import io.kinference.algorithms.completion.utils.JSON
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import java.io.File

sealed class ModelLoader {
    companion object {
        fun deserializeVocabulary(text: String): Map<String, Int> {
            return JSON.parse(MapSerializer(String.serializer(), Int.serializer()), text)
        }

        fun deserializeMerges(text: String): List<Pair<String, String>> {
            return text.lines().filterNot { it.startsWith("#") || it.isBlank() }.map { it.split(" ") }.map { (left, right) -> left to right }
        }
    }

    abstract fun getModel(): ByteArray
    abstract fun getMerges(): List<Pair<String, String>>
    abstract fun getVocabulary(): Map<String, Int>

    class FileModelLoader(model: File, vocabulary: File, merges: File) :
        CustomModelLoader({ model.readBytes() }, { vocabulary.readText() }, { merges.readText() })

    open class CustomModelLoader(val model: () -> ByteArray, val vocabulary: () -> String, val merges: () -> String) : ModelLoader() {
        override fun getModel(): ByteArray = model()
        override fun getVocabulary() = deserializeVocabulary(vocabulary())
        override fun getMerges() = deserializeMerges(merges())
    }
}
