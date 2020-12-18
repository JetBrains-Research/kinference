package io.kinference.algorithms.gec.tokenizer

import java.nio.file.Path
import io.kinference.ndarray.arrays.IntNDArray

class BertTokenizer(
    val vocabPath: Path,
    doLowerCase: Boolean = true,
    val doBasicTokenize: Boolean = true,
    unkToken: String = "[UNK]",
    sepToken: String = "[SEP]",
    padToken: String = "[PAD]",
    clsToken: String = "[CLS]",
    maskToken: String = "[MASK]",
    val tokenizeChineseChars: Boolean = true,
) : PreTrainedTokenizer(doLowerCase = doLowerCase,
    unkToken = unkToken, sepToken = sepToken,
    padToken = padToken, clsToken = clsToken, maskToken = maskToken) {

    /**
     * BertTokenizer is implementation of transformers BertTokenizer
     * [vocabPath] - for now is local vocab path
     * [doLowerCase] - boolean value for cased and uncased variants
     * [tokenizeChineseChars] - boolean value for now not used
     */

    private val vocab = load_vocab()
    private val idsToToken = vocab.entries.associate { it.value to it.key }

    private val wordPieceTokenizer = WordPieceTokenizer(vocab, unkToken)
    private val basicTokenizer = BasicTokenizer(doLowerCase, false)

    override val vocabSize: Int = vocab.size

    fun load_vocab(): Map<String, Int> {
        val tmp = vocabPath.toFile().readLines()
        return tmp.withIndex().map { (value, key) -> key to value }.toMap()
    }

    override fun convertIdToTokenOnToken(id: Int): String {
        return idsToToken.getOrDefault(id, unkToken)
    }

    override fun convertTokenToIdOnToken(token: String): Int {
        return vocab.getOrDefault(token, vocab.getValue(unkToken))
    }

    override fun tokenizeText(text: String): List<String> {
        val splitTokens = ArrayList<String>()
        if (doBasicTokenize) {
            for (token in basicTokenizer.tokenize(text)) {
                splitTokens += wordPieceTokenizer.tokenize(token)
            }
        } else {
            splitTokens += wordPieceTokenizer.tokenize(text)
        }
        return splitTokens
    }

    override fun encode(text: String, addSpecialTokens: Boolean): List<Int> {
        val tokenizedText = tokenize(text)
        val encoded = ArrayList<Int>()
        if (addSpecialTokens) {
            encoded.add(vocab.getOrDefault(clsToken, vocab.get(unkToken)!!))
        }
        for (token in tokenizedText) {
            encoded.add(vocab.getOrDefault(token, vocab.get(unkToken)!!))
        }
        if (addSpecialTokens) {
            encoded.add(vocab.getOrDefault(sepToken, vocab.get(unkToken)!!))
        }
        return encoded
    }

    override fun decode(ids: List<Int>): String {
        val tokens = ArrayList<String>()
        ids.forEach { tokens.add(idsToToken.getOrDefault(it, unkToken)) }
        return tokens.joinToString(" ")
    }

    fun encodeBatch(texts: List<String>, addSpecialTokens: Boolean): List<List<Int>> {
        val result = ArrayList<List<Int>>()
        for (text in texts) {
            result.add(encode(text, addSpecialTokens))
        }
        return result
    }

    fun encode2tensor(text: String, addSpecialTokens: Boolean): TokenizedInputs {
        val encoded = encode(text, addSpecialTokens)
        val inputIds = IntNDArray(
            shape = IntArray(size = 2, init = { i: Int -> if (i == 0) 1 else encoded.size }),
            init = { i: Int -> encoded[i] })
        return TokenizedInputs(inputsIds = inputIds, padTokenId = vocab.get(padToken)!!)
    }

    fun encodeBatch2tensor(texts: List<String>, addSpecialTokens: Boolean): TokenizedInputs {
        val batch = encodeBatch(texts = texts, addSpecialTokens)
        val paddedBatch = padSequences(batch)
        val innerSize = paddedBatch[0].size
        val inputIds = IntNDArray(shape = IntArray(size = 2, init = { i: Int ->
            if (i == 0)
                paddedBatch.size
            else innerSize
        }),
            init = { i: Int -> paddedBatch[i / innerSize][i % innerSize] })
        return TokenizedInputs(inputsIds = inputIds, padTokenId = vocab.get(padToken)!!)
    }

    private fun padSequences(sequences: List<List<Int>>): List<List<Int>> {
        val maxLen = sequences.map { it.size }.maxOrNull()!!
        val paddedSequences = ArrayList<List<Int>>()
        for (seq in sequences) {
            val paddedSentence = ArrayList<Int>()
            for (item in seq) {
                paddedSentence.add(item)
            }
            if (paddedSentence.size < maxLen) {
                for (i in 0 until maxLen - paddedSentence.size) {
                    paddedSentence.add(vocab.getOrDefault(padToken, vocab.get(unkToken)!!))
                }
            }
            paddedSequences.add(paddedSentence)
        }
        return paddedSequences
    }
}
