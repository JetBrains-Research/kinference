package io.kinference.algorithms.gec.preprocessing

import io.kinference.algorithms.gec.tokenizer.AutoTokenizer

/**
 * Interface for realization of text processors
 */
interface TextProcessor {
    fun encodeAsIds(text: String): List<Int>

    fun decodeIds(ids: List<Int>): String

    fun encodeAsTokens(text: String): List<String>

    fun decodeTokens(tokens: List<String>): String {
        return tokens.joinToString(" ")
    }

    val bosId: Int

    val eosId: Int

    val padId: Int

    val maskId: Int

    val unkId: Int

    val sepId: Int

    val clsId: Int

    val numWords: Int
}

/**
 * TextProcessor from transformer tokenizer for text processing
 */
class TransformersTextprocessor(val modelNameOrPath: String) : TextProcessor {
    val tokenizer = AutoTokenizer.fromPretrained(modelNameOrPath)

    override fun encodeAsIds(text: String): List<Int> {
        return tokenizer.encode(text, false)
    }

    override fun decodeIds(ids: List<Int>): String {
        return tokenizer.decode(ids)
    }

    override fun encodeAsTokens(text: String): List<String> {
        return tokenizer.tokenize(text)
    }

    override fun decodeTokens(tokens: List<String>): String {
        return tokenizer.covertTokensToString(tokens)
    }

    override val bosId: Int
        get() = tokenizer.clsId

    override val eosId: Int
        get() = tokenizer.sepId

    override val padId: Int
        get() = tokenizer.padId

    override val maskId: Int
        get() = tokenizer.maskId

    override val unkId: Int
        get() = tokenizer.unkId

    override val sepId: Int
        get() = tokenizer.sepId

    override val clsId: Int
        get() = tokenizer.clsId

    override val numWords: Int
        get() = tokenizer.vocabSize
}


