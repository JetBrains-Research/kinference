package io.kinference.algorithms.gec.tokenizer

/**
 * Base interface for all Tokenizers which trying to mimic transformers Tokenizers
 */

abstract class PreTrainedTokenizer(val doLowerCase: Boolean,
                                   val unkToken: String,
                                   val sepToken: String,
                                   val padToken: String,
                                   val clsToken: String,
                                   val maskToken: String) {
    /**
     * Base class for Tokenizer which using the vocabulary should implement following parameters
     * [vocabSize] - length of Tokenizer vocabulary
     * [doLowerCase] - boolean value for cased and uncased tokenizers
     * [unkToken] - token which denotes unknown word
     * [sepToken] - token which denotes end of sentence/end of segment
     * [padToken] - token which denotes padding token for batch processing
     * [clsToken] - token which denotes begin of sentence/begin of segment
     * [maskToken] - token which denotes masking token for masked language modeling (not use in our approach)
     */

    abstract val vocabSize: Int

    val unkId: Int by lazy { convertTokenToIdOnToken(unkToken) }
    val sepId: Int by lazy { convertTokenToIdOnToken(sepToken) }
    val padId: Int by lazy { convertTokenToIdOnToken(padToken) }
    val clsId: Int by lazy { convertTokenToIdOnToken(clsToken) }
    val maskId: Int by lazy { convertTokenToIdOnToken(maskToken) }


    fun length(): Int {
        return vocabSize
    }

    abstract fun tokenizeText(text: String): List<String>

    abstract fun convertTokenToIdOnToken(token: String): Int

    abstract fun convertIdToTokenOnToken(id: Int): String

    abstract fun encode(text: String, addSpecialTokens: Boolean): List<Int>

    abstract fun decode(ids: List<Int>): String

    fun splitOnToken(tok: String, text: String): List<String> {
        val result = mutableListOf<String>()
        val split_text = text.split(tok)

        for ((i, subText) in split_text.withIndex()) {
            var mSubText: String = subText
            if (i < split_text.size - 1) {
                mSubText = mSubText.trimEnd()
            }
            if (i > 0) {
                mSubText = mSubText.trimStart()
            }
            if (i == 0 && !mSubText.isNullOrEmpty()) {
                result.add(tok)
            } else if (i == mSubText.length - 1) {
                if (!mSubText.isNullOrEmpty()) {
                    result.add(mSubText)
                }
            } else {
                if (!mSubText.isNullOrEmpty()) {
                    result.add(mSubText)
                }
                result.add(tok)
            }
        }
        return result
    }

    fun splitOnTokens(tokList: List<String>, text: String): List<String> {
        if (text.trimEnd().trimStart().isNullOrEmpty()) {
            return emptyList()
        }
        if (tokList.isNullOrEmpty()) {
            return tokenizeText(text)
        }
        var textList = mutableListOf(text)

        for (tok in tokList) {
            val tokenizedText = textList.map { subText -> splitOnToken(tok, subText) }
            textList = tokenizedText.flatten() as MutableList<String>
        }
        val result = textList.map { token: String -> tokenizeText(token) }
        return result.flatten()

    }

    fun tokenize(text: String): List<String> {
        val mText: String = if (doLowerCase) {
            text.toLowerCase()
        } else {
            text
        }
        return splitOnTokens(listOf(), mText)
    }

    fun convertTokensToIds(tokens: List<String>): List<Int> {
        if (tokens.isNullOrEmpty()) {
            return emptyList()
        }

        return tokens.map { token -> convertTokenToIdOnToken(token) }
    }

    fun convertIdsToTokens(ids: List<Int>): List<String> {
        return ids.map { id -> convertIdToTokenOnToken(id) }
    }

    fun covertTokensToString(tokens: List<String>): String {
        return tokens.joinToString(" ")
    }
}
