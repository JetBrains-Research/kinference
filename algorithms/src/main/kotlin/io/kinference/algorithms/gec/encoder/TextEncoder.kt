package io.kinference.algorithms.gec.encoder

/**
 * Encoder and decoder of text used by DL models
 */
interface TextEncoder {
    fun encodeAsTokens(text: String): List<String>
    fun decodeFromTokens(tokens: List<String>): String

    fun encodeAsIds(text: String, withSpecialTokens: Boolean): List<Int>
    fun decodeFromIds(ids: List<Int>): String


    fun encodeAsTokens(texts: List<String>) = texts.map { encodeAsTokens(it) }
    fun encodeAsIds(texts: List<String>, withSpecialTokens: Boolean) = texts.map { encodeAsIds(it, withSpecialTokens) }
}

