package io.kinference.algorithms.gec.tokenizer

/**
 * Basic interface for tokenization of text to words or sentences
 */
interface Tokenizer {
    /** Performs tokenization of text */
    fun tokenize(text: String): List<String>
}
