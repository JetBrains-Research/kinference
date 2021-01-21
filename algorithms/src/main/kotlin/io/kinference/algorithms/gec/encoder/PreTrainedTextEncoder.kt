package io.kinference.algorithms.gec.encoder

/**
 * Base class for text encoder that is using some vocabulary
 *
 * @param toLowerCase boolean value for cased and uncased tokenizers
 * @param unkToken token which denotes unknown word
 * @param sepToken token which denotes end of sentence/end of segment
 * @param padToken token which denotes padding token for batch processing
 * @param clsToken token which denotes begin of sentence/begin of segment
 * @param maskToken token which denotes masking token for masked language modeling (not use in our approach)
 */

abstract class PreTrainedTextEncoder(
    val toLowerCase: Boolean,
    val unkToken: String,
    val sepToken: String,
    val padToken: String,
    val clsToken: String,
    val maskToken: String
) : TextEncoder {

    abstract val vocabSize: Int

    val unkId: Int by lazy { convertTokenToIdOnToken(unkToken) }
    val sepId: Int by lazy { convertTokenToIdOnToken(sepToken) }
    val padId: Int by lazy { convertTokenToIdOnToken(padToken) }
    val clsId: Int by lazy { convertTokenToIdOnToken(clsToken) }
    val maskId: Int by lazy { convertTokenToIdOnToken(maskToken) }

    override fun encodeAsTokens(text: String): List<String> {
        val mText: String = if (toLowerCase) {
            text.toLowerCase()
        } else {
            text
        }
        return splitByTokens(emptyList(), mText)
    }

    override fun decodeFromTokens(tokens: List<String>): String {
        return tokens.joinToString(" ")
    }

    protected abstract fun tokenizeText(text: String): List<String>

    protected abstract fun convertTokenToIdOnToken(token: String): Int

    protected abstract fun convertIdToTokenOnToken(id: Int): String

    private fun splitByToken(token: String, text: String): List<String> {
        val result = ArrayList<String>()
        val split = text.split(token)

        for ((i, subText) in split.withIndex()) {
            var mSubText: String = subText
            if (i < split.size - 1) {
                mSubText = mSubText.trimEnd()
            }
            if (i > 0) {
                mSubText = mSubText.trimStart()
            }
            if (i == 0 && mSubText.isNotEmpty()) {
                result.add(token)
            } else if (i == mSubText.length - 1) {
                if (mSubText.isNotEmpty()) {
                    result.add(mSubText)
                }
            } else {
                if (mSubText.isNotEmpty()) {
                    result.add(mSubText)
                }
                result.add(token)
            }
        }
        return result
    }

    private fun splitByTokens(tokens: List<String>, text: String): List<String> {
        if (text.isBlank()) return emptyList()
        if (tokens.isEmpty()) return tokenizeText(text)

        var textList = listOf(text)

        for (token in tokens) {
            textList = textList.flatMap { subText -> splitByToken(token, subText) }
        }
        return textList.flatMap { token: String -> tokenizeText(token) }
    }
}
