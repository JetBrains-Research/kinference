package io.kinference.algorithms.gec.preprocessing

import io.kinference.algorithms.gec.encoder.BertTextEncoder
import java.io.File

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
class TransformersTextProcessor(modelPath: String) : TextProcessor {
    private val tokenizer = BertTextEncoder(vocabularyFile = File(modelPath))

    override fun encodeAsIds(text: String): List<Int> = tokenizer.encodeAsIds(text, false)

    override fun decodeIds(ids: List<Int>): String = tokenizer.decodeFromIds(ids)

    override fun encodeAsTokens(text: String): List<String> = tokenizer.encodeAsTokens(text)

    override fun decodeTokens(tokens: List<String>): String = tokenizer.decodeFromTokens(tokens)

    /** Begin of sentence (BOS) token id */
    override val bosId: Int
        get() = tokenizer.clsId

    /** End of sentence (EOS) token id */
    override val eosId: Int
        get() = tokenizer.sepId

    /** Padding token id */
    override val padId: Int
        get() = tokenizer.padId

    /** Mask token id */
    override val maskId: Int
        get() = tokenizer.maskId

    /** Unknown token id */
    override val unkId: Int
        get() = tokenizer.unkId

    /** Separative token id (For Transformer-like architecture) */
    override val sepId: Int
        get() = tokenizer.sepId

    /** CLS token id (For Transformer-like architecture) */
    override val clsId: Int
        get() = tokenizer.clsId

    /** Total number of words in tokenizer vocabulary */
    override val numWords: Int
        get() = tokenizer.vocabSize
}


