package io.kinference.algorithms.gec.corrector.correction

import io.kinference.algorithms.gec.GECTag
import io.kinference.algorithms.gec.changes.TokenChangesGenerator
import io.kinference.algorithms.gec.classifier.GECClassifier
import io.kinference.algorithms.gec.corrector.GECTagger
import io.kinference.algorithms.gec.encoder.PreTrainedTextEncoder
import io.kinference.algorithms.gec.preprocessing.VerbsFormVocabulary
import io.kinference.algorithms.gec.tokenizer.TokenRange
import io.kinference.algorithms.gec.utils.*
import kotlin.math.abs

/**
 * SentenceCorrections represents corrections generated during inference of model
 *
 * Those corrections are used internally in model and should not be used by end-users.
 *
 * Instead, for end-users [TextCorrection] proposed
 *
 * In python code it is called SentenceCorrection
 */
data class SentenceCorrections(
    val sentId: Int,
    val sent: String,
    val tokens: List<GECToken>,
    val corrections: HashMap<String, TokenCorrection> = HashMap(),
    var isCorrect: Boolean = false
) {

    data class TokenCorrection(val tag: String, val errorClass: String, var changedTokens: List<GECToken>)

    data class GECToken(
        val text: String, val encoded: List<Int>, val range: TokenRange,
        val isFirst: Boolean, val isUsed: Boolean
    ) {

        /**
         * Information about token
         */

        var position: String = "none"

        fun initialTokenPosition(): Int {
            return position.split('.', limit = 2)[0].toInt()
        }

        fun isStartToken(): Boolean {
            return text == "\$START"
        }
    }


    init {
        tokens.forEachIndexed { pos, t -> t.position = pos.toString() }
    }

    fun addTokenToCorrections(
        tokenSentence: List<GECToken>, taggedSentence: GECTagger.TagSentObject,
        encoder: PreTrainedTextEncoder, verbsFormVocabulary: VerbsFormVocabulary
    ) {
        require(taggedSentence.tokens == tokenSentence.map { it.text })

        val changesGenerator = TokenChangesGenerator(tokenSentence, taggedSentence.tags, verbsFormVocabulary = verbsFormVocabulary)
        val changesList = changesGenerator.generateTokenChanges()

        for (idx in changesList.indices) {
            val token = tokenSentence[idx]
            val tag = taggedSentence.tags[idx]
            val changes = changesList[idx] ?: continue

            val changedTokens = ArrayList<GECToken>()
            if (changes.replacement == "") {
                changedTokens.add(
                    GECToken(
                        text = "", range = token.range,
                        encoded = emptyList(), isUsed = false, isFirst = token.isFirst
                    )
                )
            } else {
                val tokenRangeList = TokenRange.findTokensInText(
                    text = changes.replacement,
                    tokens = changes.tokenizedReplacement!!, textWithSpace = token.range.withSpace
                )

                val start = tokenSentence[idx].range.start
                val end = tokenSentence[idx + changes.usedTokensNum - 1].range.endExclusive

                for (index in changes.tokenizedReplacement!!.indices) {
                    changedTokens.add(
                        GECToken(
                            text = changes.tokenizedReplacement!![index],
                            range = TokenRange(start = start, endExclusive = end, withSpace = tokenRangeList[index].withSpace),
                            encoded = encoder.encodeAsIds(changes.tokenizedReplacement!![index], false),
                            isUsed = token.isUsed, isFirst = false
                        )
                    )
                }
            }
            addTokenCorrection(
                token = token,
                correction = TokenCorrection(tag = tag, errorClass = GECClassifier.classifyError(tag), changedTokens = changedTokens)
            )
        }

    }

    private fun addTokenCorrection(token: GECToken, correction: TokenCorrection) {
        val changedTokens = ArrayList<GECToken>()
        val tokenPos = token.position
        for ((pos, t) in correction.changedTokens.withIndex()) {
            t.position = "${tokenPos}.${pos}"

            changedTokens.add(t)
        }
        correction.changedTokens = changedTokens
        corrections[tokenPos] = correction
    }

    fun toTextCorrections() = getTokensToMerge().map { constructMergedCorrection(it) }

    fun toCorrectedTokenSentence() = tokens.flatMap { parseTokenCorrection(it) }

    private fun parseTokenCorrection(token: GECToken): List<GECToken> {
        return if (token.position in corrections) {
            val cTokens = ArrayList<GECToken>()

            for (cToken in corrections[token.position]!!.changedTokens) {
                cTokens += parseTokenCorrection(cToken)
            }

            cTokens
        } else {
            listOf(token)
        }
    }

    private fun getTokensToMerge(): List<List<GECToken>> {
        val tokensToCorrect = ArrayList<GECToken>()

        for (token in tokens.filter { it.position in corrections }) {
            if (corrections[token.position]!!.changedTokens.size != 1 || corrections[token.position]!!.changedTokens[0].text != token.text) {
                tokensToCorrect.add(token)
            }
        }

        if (tokensToCorrect.isEmpty()) return emptyList()

        val tokensToMerge = ArrayList<List<GECToken>>()
        var currentMerge = mutableListOf(tokensToCorrect[0])

        val from = 1
        val to = tokensToCorrect.lastIndex
        if (to >= from) {
            for (cur in tokensToCorrect.subList(fromIndex = 1, toIndex = tokensToCorrect.lastIndex + 1)) {
                if (isNeighbour(currentMerge.last(), cur)) {
                    val allTokens = getInBetweenTokens(currentMerge.last(), cur)
                    currentMerge.addAll(allTokens)
                    currentMerge.add(cur)
                } else {
                    require(currentMerge.size > 0)
                    tokensToMerge.add(currentMerge)
                    currentMerge = mutableListOf(cur)
                }
            }
        }

        if (currentMerge.isNotEmpty()) tokensToMerge.add(currentMerge)

        return tokensToMerge
    }

    private fun isNeighbour(one: GECToken, two: GECToken, radius: Int = 3): Boolean {
        require(radius >= 1) { "Radius is expected to be >= 1, but was $radius" }
        return abs(two.initialTokenPosition() - one.initialTokenPosition()) <= radius
    }

    private fun getInBetweenTokens(one: GECToken, two: GECToken): List<GECToken> {
        require(one.initialTokenPosition() <= two.initialTokenPosition())
        return tokens.subList(fromIndex = one.initialTokenPosition() + 1, toIndex = two.initialTokenPosition())
    }

    private fun constructMergedCorrection(tokens: List<GECToken>): TextCorrection {
        return TextCorrection(
            errorRange = createErrorRange(tokens),
            underlineRange = createUnderlineRange(tokens),
            message = createMessage(tokens),
            replacement = createReplacement(tokens)
        )
    }

    private fun createErrorRange(tokens: List<GECToken>): IntRange {
        val tag: String
        var range: IntRange
        val word: String
        val withSpace: Boolean

        if (tokens.size == 1) {
            val token = tokens[0]
            require(token.position in corrections)

            tag = corrections[token.position]!!.tag
            range = IntRange(token.range.start, token.range.endExclusive)
            word = token.text
            withSpace = token.range.withSpace
        } else {
            val startToken = tokens[0]
            val endToken = tokens.last()

            range = IntRange(startToken.range.start, endToken.range.endExclusive)
            require(startToken.position in corrections && endToken.position in corrections)
            val isDelete = ArrayList<Boolean>()

            for (token in tokens) {
                if (corrections.containsKey(token.position) &&
                    corrections[token.position]!!.tag == GECTag.DELETE.value
                ) {
                    isDelete.add(true)
                } else {
                    isDelete.add(false)
                }
            }
            tag = if (isDelete.all { it }) GECTag.DELETE.value else GECTag.KEEP.value
            word = endToken.text
            withSpace = endToken.range.withSpace
        }

        if (tag == GECTag.DELETE.value) {
            range = range.withRightExpand(1)
        }

        return if (tag == GECTag.DELETE.value && !withSpace && word != " ") {
            range
        } else {
            range.withRightExpand(-1)
        }
    }

    private fun createUnderlineRange(tokens: List<GECToken>): IntRange = IntRange(tokens.first().range.start, tokens.last().range.endExclusive - 1)

    private fun createMessage(tokens: List<GECToken>): String {
        return if (tokens.size == 1) {
            require(tokens[0].position in corrections)
            corrections[tokens[0].position]!!.errorClass
        } else {
            "Complex error"
        }
    }

    private fun createReplacement(tokens: List<GECToken>): String {
        var replacement = ""
        for (token in tokens) {
            var tokenReplacement = ""
            if (token.position in corrections) {
                for (cToken in parseTokenCorrection(token)) {
                    tokenReplacement = updateReplacement(tokenReplacement, cToken.text, withSpace = cToken.range.withSpace)
                }
            } else {
                tokenReplacement = token.text
            }
            replacement = updateReplacement(replacement, tokenReplacement, withSpace = token.range.withSpace)
        }
        return replacement
    }

    private fun updateReplacement(current: String, token: String, withSpace: Boolean): String {
        return if (current != "" && token != "") {
            if (withSpace) {
                "$current $token"
            } else {
                current + token
            }
        } else {
            current + token
        }
    }
}
