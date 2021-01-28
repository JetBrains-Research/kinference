package io.kinference.algorithms.gec.encoder

import io.kinference.algorithms.gec.tokenizer.subword.WordPieceTokenizer
import io.kinference.algorithms.gec.tokenizer.word.BasicWordTokenizer
import java.io.File


/**
 * BertTexEncoder is implementation of transformers BertTokenizer
 * @param vocabularyFile path to vocabulary
 * @param toLowerCase boolean value for cased and uncased variants
 */
class BertTextEncoder(
    vocabularyFile: String,
    toLowerCase: Boolean = true,
    private val doBasicTokenize: Boolean = true,
    unkToken: String = "[UNK]",
    sepToken: String = "[SEP]",
    padToken: String = "[PAD]",
    clsToken: String = "[CLS]",
    maskToken: String = "[MASK]"
) : PreTrainedTextEncoder(
    toLowerCase = toLowerCase,
    unkToken = unkToken, sepToken = sepToken,
    padToken = padToken, clsToken = clsToken, maskToken = maskToken
) {

    private val vocabulary = vocabularyFile.lines().filter { it.isNotBlank() }.withIndex().associate { (value, key) -> key to value }
    private val idsToToken = vocabulary.entries.associate { it.value to it.key }

    private val wordPieceTokenizer = WordPieceTokenizer(vocabulary, unkToken)
    private val basicWordTokenizer = BasicWordTokenizer(toLowerCase, shouldNormalizeAccentLetters = false)

    override val vocabSize: Int = vocabulary.size

    override fun convertIdToTokenOnToken(id: Int): String = idsToToken.getOrDefault(id, unkToken)

    override fun convertTokenToIdOnToken(token: String): Int = vocabulary.getOrDefault(token, vocabulary.getValue(unkToken))

    override fun tokenizeText(text: String): List<String> {
        val splitTokens = ArrayList<String>()
        if (doBasicTokenize) {
            for (token in basicWordTokenizer.tokenize(text)) {
                splitTokens += wordPieceTokenizer.tokenize(token)
            }
        } else {
            splitTokens += wordPieceTokenizer.tokenize(text)
        }
        return splitTokens
    }

    override fun encodeAsIds(text: String, withSpecialTokens: Boolean): List<Int> {
        val tokenizedText = encodeAsTokens(text)
        val encoded = ArrayList<Int>()
        if (withSpecialTokens) {
            encoded.add(vocabulary.getOrDefault(clsToken, vocabulary[unkToken]!!))
        }
        for (token in tokenizedText) {
            encoded.add(vocabulary.getOrDefault(token, vocabulary[unkToken]!!))
        }
        if (withSpecialTokens) {
            encoded.add(vocabulary.getOrDefault(sepToken, vocabulary[unkToken]!!))
        }
        return encoded
    }

    override fun decodeFromIds(ids: List<Int>): String {
        val tokens = ArrayList<String>()
        for (it in ids) {
            tokens.add(idsToToken.getOrDefault(it, unkToken))
        }
        return tokens.joinToString(" ")
    }
}
