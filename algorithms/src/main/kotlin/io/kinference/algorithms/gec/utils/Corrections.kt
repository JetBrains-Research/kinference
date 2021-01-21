package io.kinference.algorithms.gec.utils

import io.kinference.algorithms.gec.encoder.PreTrainedTextEncoder
import io.kinference.algorithms.gec.GECTag
import io.kinference.algorithms.gec.preprocessing.VerbsFormVocabulary
import io.kinference.algorithms.gec.utils.GECToken.TokenRange
import kotlin.math.abs

/**
 * Correction of error in text -- output of GEC model.
 *
 * @param errorRange is a range for which replacement is prepared
 * @param underlineRange is a range that should be highlighted
 * @param replacement is a suggestion of fix
 * @param message is a description of fix
 */
data class TextCorrection(val errorRange: IntRange,
                          val underlineRange: IntRange,
                          val replacement: String,
                          val message: String) {
    /** Apply correction to this sentence */
    fun apply(sentence: String): String = sentence.replaceRange(errorRange, replacement)
}

/**
 * Sentence piece class which change tokens using corrections
 */
data class SentenceCorrections(val sentId: Int,
                               val sent: String,
                               var tokens: List<GECToken>,
                               val corrections: HashMap<String, TokenCorrection> = HashMap(),
                               var isCorrect: Boolean = false) {
    init {
        tokens.forEachIndexed() { pos, t -> t.position = pos.toString() }
    }

    fun addTokenToCorrections(tokenSentence: List<GECToken>, taggedSentence: TagSentObject,
                              encoder: PreTrainedTextEncoder, verbsFormVocabulary: VerbsFormVocabulary) {
        assert(taggedSentence.tokens == tokenSentence.map { it.text })
        val changesGenerator = TokenChangesGenerator(tokenSentence, taggedSentence.tags, verbsFormVocabulary = verbsFormVocabulary)
        val changesList = changesGenerator.generateTokenChanges()

        for (idx in changesList.indices) {
            val token = tokenSentence[idx]
            val tag = taggedSentence.tags[idx]
            val changes = changesList[idx] ?: continue

            val changedTokens = ArrayList<GECToken>()
            if (changes.replacement == "") {
//                token.text = ""
//                token.encodedData = emptyList()
//                token.isUsed = false

                changedTokens.add(GECToken(text = "", range = token.range,
                    encoded = emptyList(), isUsed = false, isFirst = token.isFirst))
            } else {
                val tokenRangeList = calculateTokensBordersAndWithSpaces(text = changes.replacement,
                    tokens = changes.tokenizedReplacement!!, textWithSpace = token.range.withSpace)

                val withSpaces = tokenRangeList.map { it.withSpace }

                val start = tokenSentence[idx].range.start
                val end = tokenSentence[idx + changes.usedTokensNum - 1].range.end

                for (index in changes.tokenizedReplacement!!.indices) {
                    changedTokens.add(GECToken(text = changes.tokenizedReplacement!![index],
                        range = TokenRange(start = start, end = end, withSpace = withSpaces[index]),
                        encoded = encoder.encodeAsIds(changes.tokenizedReplacement!![index], false),
                        isUsed = token.isUsed, isFirst = false))
                }
            }
            addTokenCorrection(token = token, correction = TokenCorrection(
                tag = tag, errorClass = createMessageBasedOnTag(tag),
                changedTokens = changedTokens))
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

    fun toTextCorrections(): List<TextCorrection> {
        val tokensToMerge = getTokensToMerge()
        val result = ArrayList<TextCorrection>()
        for (tokens in tokensToMerge) {
            result.add(constructMergedCorrection(tokens))
        }
        return result
    }

    fun toCorrectedTokenSentence(): List<GECToken> {
        val correctedSentence = ArrayList<GECToken>()
        for (token in tokens) {
            correctedSentence += parseTokenCorrection(token)
        }
        return correctedSentence
    }

    private fun parseTokenCorrection(token: GECToken): List<GECToken> {
        if (corrections.contains(token.position)) {
            val ctokens = ArrayList<GECToken>()

            for (cToken in corrections[token.position]!!.changedTokens) {
                ctokens += parseTokenCorrection(cToken)
            }
            return ctokens
        } else {
            return listOf(token)
        }
    }

    private fun parseTokenCorrectionToString(token: GECToken): List<String> {
        return parseTokenCorrection(token).map { it.text }
    }

    private fun getTokensToMerge(): List<List<GECToken>> {
        val tokensToCorrect = ArrayList<GECToken>()

        for (token in tokens) {
            if (token.position in corrections) {
                if (corrections[token.position]!!.changedTokens.size != 1 || corrections[token.position]!!.changedTokens[0].text != token.text) {
                    tokensToCorrect.add(token)
                }
            }
        }

        if (tokensToCorrect.isNullOrEmpty()) {
            return emptyList()
        }

        val tokensToMerge = ArrayList<List<GECToken>>()
        var currentMerge = mutableListOf(tokensToCorrect[0])

        val from = 1
        val to = tokensToCorrect.lastIndex
        if (to >= from) {
            for (cur in tokensToCorrect.subList(fromIndex = 1, toIndex = tokensToCorrect.lastIndex + 1)) {
                if (isNeighbour(currentMerge.last(), cur)) {
                    val alltokens = getInBetweenTokens(currentMerge.last(), cur)
                    currentMerge.addAll(alltokens)
                    currentMerge.add(cur)
                } else {
                    assert(currentMerge.size > 0)
                    tokensToMerge.add(currentMerge)
                    currentMerge = mutableListOf(cur)
                }
            }
        }

        if (currentMerge.isNotEmpty()) {
            tokensToMerge.add(currentMerge)
        }
        return tokensToMerge
    }

    private fun isNeighbour(one: GECToken, two: GECToken, radius: Int = 3): Boolean {
        assert(radius >= 1)
        return abs(two.initialTokenPosition() - one.initialTokenPosition()) <= radius
    }

    private fun getInBetweenTokens(one: GECToken, two: GECToken): List<GECToken> {
        assert(one.initialTokenPosition() <= two.initialTokenPosition())
        return tokens.subList(fromIndex = one.initialTokenPosition() + 1,
            toIndex = two.initialTokenPosition())
    }

    private fun constructMergedCorrection(tokens: List<GECToken>): TextCorrection {
        return TextCorrection(errorRange = createErrorRange(tokens),
            underlineRange = createUnderlineRange(tokens),
            message = createMessage(tokens),
            replacement = createReplacement(tokens))
    }

    private fun createErrorRange(tokens: List<GECToken>): IntRange {
        val tag: String
        var startEnd: Pair<Int, Int>
        val word: String
        val withSpace: Boolean

        if (tokens.size == 1) {
            val token = tokens[0]
            assert(token.position in corrections)
            tag = corrections[token.position]!!.tag
            startEnd = Pair(token.range.start, token.range.end)
            word = token.text
            withSpace = token.range.withSpace
        } else {
            val startToken = tokens[0]
            val endToken = tokens.last()
            startEnd = Pair(startToken.range.start, endToken.range.end)
            assert(startToken.position in corrections && endToken.position in corrections)
            val isDelete = ArrayList<Boolean>()

            for (token in tokens) {
                if (corrections.containsKey(token.position) &&
                    corrections.get(token.position)!!.tag == GECTag.DELETE.value) {
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
            startEnd = Pair(startEnd.first, startEnd.second + 1)
        }

        return if (tag == GECTag.DELETE.value && !withSpace && word != " ") {
            IntRange(startEnd.first, startEnd.second)
        } else {
            IntRange(startEnd.first, startEnd.second - 1)
        }
    }

    private fun createUnderlineRange(tokens: List<GECToken>): IntRange = IntRange(tokens.first().range.start, tokens.last().range.end - 1)

    private fun createMessage(tokens: List<GECToken>): String {
        return if (tokens.size == 1) {
            assert(tokens[0].position in corrections)
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
                    tokenReplacement = updateReplacementString(tokenReplacement, cToken.text, withSpace = cToken.range.withSpace)
                }
            } else {
                tokenReplacement = token.text
            }
            replacement = updateReplacementString(replacement, tokenReplacement, withSpace = token.range.withSpace)
        }
        return replacement
    }

    private fun updateReplacementString(current: String, token: String, withSpace: Boolean): String {
        if (current != "" && token != "") {
            if (withSpace) {
                return "$current $token"
            } else {
                return current + token
            }
        } else {
            return current + token
        }
    }
}
