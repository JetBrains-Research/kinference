package io.kinference.algorithms.gec.utils

import io.kinference.algorithms.gec.preprocessing.TransformersTextprocessor
import io.kinference.algorithms.gec.preprocessing.VerbsFormVocabulary
import io.kinference.algorithms.gec.preprocessing.TagDelete
import io.kinference.algorithms.gec.preprocessing.TagKeep
import kotlin.math.abs

/**
 * Class which implements sentence-part correction. Output of GecCorrector
 */
data class TextCorrection(val errorRange: Pair<Int, Int>,
                          val underlineRange: Pair<Int, Int>,
                          val replacement: String,
                          val massage: String)

/**
 * Sentence piece class which change tokens using corrections
 */
data class SentenceCorrections(val sentId: Int, val sent: String,
                               var tokens: List<Token>,
                               val corrections: HashMap<String, TokenCorrection> = HashMap(),
                               var isCorrect: Boolean = false) {
    init {
        tokens.forEachIndexed() { pos, t -> t.position = pos.toString() }
    }

    fun addTokenToCorrections(tokenSentence: List<Token>, taggedSentence: TagSentObject,
                              textProcessor: TransformersTextprocessor, verbsFormVocabulary: VerbsFormVocabulary) {
        assert(taggedSentence.tokens == tokenSentence.map { it.text })
        val changesGenerator = TokenChangesGenerator(tokenSentence, taggedSentence.tags, verbsFormVocabulary = verbsFormVocabulary)
        val changesList = changesGenerator.generateTokenChanges()

        for (idx in changesList.indices) {
            val token = tokenSentence[idx]
            val tag = taggedSentence.tags[idx]
            val changes = changesList[idx] ?: continue

            val changedTokens = ArrayList<Token>()
            if (changes!!.replacement == "") {
//                token.text = ""
//                token.encodedData = emptyList()
//                token.isUsed = false

                changedTokens.add(Token(text = "", tokenRange = token.tokenRange,
                    encodedData = emptyList(), isUsed = false, isFirst = token.isFirst))
            } else {
                val tokenRangeList = calculateTokensBordersAndWithSpaces(text = changes.replacement,
                    tokens = changes.tokenizedReplacement!!, textWithSpace = token.tokenRange.withSpace)

                val withSpaces = tokenRangeList.map { it.withSpace }

                val start = tokenSentence[idx].tokenRange.start
                val end = tokenSentence[idx + changes.usedTokensNum - 1].tokenRange.end

                for (index in changes.tokenizedReplacement!!.indices) {
                    changedTokens.add(Token(text = changes.tokenizedReplacement!![index],
                        tokenRange = TokenRange(start = start, end = end, withSpace = withSpaces[index]),
                        encodedData = textProcessor.encodeAsIds(changes.tokenizedReplacement!![index]),
                        isUsed = token.isUsed, isFirst = false))
                }
            }
            addTokenCorrection(token = token, correction = TokenCorrection(
                tag = tag, errorClass = createMessageBasedOnTag(tag),
                changedTokens = changedTokens))
        }

    }

    private fun addTokenCorrection(token: Token, correction: TokenCorrection) {
        val changedTokens = ArrayList<Token>()
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
        if (tokensToMerge != null) {
            val result = ArrayList<TextCorrection>()
            for (tokens in tokensToMerge) {
                result.add(constructMergedCorrection(tokens))
            }
            return result
        } else {
            return emptyList()
        }
    }

    fun toCorrectedTokenSentence(): List<Token> {
        val correctedSentence = ArrayList<Token>()
        for (token in tokens) {
            correctedSentence += parseTokenCorrection(token)
        }
        return correctedSentence
    }

    private fun parseTokenCorrection(token: Token): List<Token> {
        if (corrections.contains(token.position)) {
            val ctokens = ArrayList<Token>()

            for (cToken in corrections[token.position]!!.changedTokens) {
                ctokens += parseTokenCorrection(cToken)
            }
            return ctokens
        } else {
            return listOf(token)
        }
    }

    private fun parseTokenCorrectionToString(token: Token): List<String> {
        return parseTokenCorrection(token).map { it.text }
    }

    private fun getTokensToMerge(): List<List<Token>> {
        val tokensToCorrect = ArrayList<Token>()

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

        val tokensToMerge = ArrayList<List<Token>>()
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

    private fun isNeighbour(one: Token, two: Token, radius: Int = 3): Boolean {
        assert(radius >= 1)
        return abs(two.initialTokenPosition() - one.initialTokenPosition()) <= radius
    }

    private fun getInBetweenTokens(one: Token, two: Token): List<Token> {
        assert(one.initialTokenPosition() <= two.initialTokenPosition())
        return tokens.subList(fromIndex = one.initialTokenPosition() + 1,
            toIndex = two.initialTokenPosition())
    }

    private fun constructMergedCorrection(tokens: List<Token>): TextCorrection {
        return TextCorrection(errorRange = createErrorRange(tokens),
            underlineRange = createUnderlineRange(tokens),
            massage = createMessage(tokens),
            replacement = createReplacement(tokens))
    }

    private fun createErrorRange(tokens: List<Token>): Pair<Int, Int> {
        val tag: String
        var startEnd: Pair<Int, Int>
        val word: String
        val withSpace: Boolean

        if (tokens.size == 1) {
            val token = tokens[0]
            assert(token.position in corrections)
            tag = corrections[token.position]!!.tag
            startEnd = Pair(token.tokenRange.start, token.tokenRange.end)
            word = token.text
            withSpace = token.tokenRange.withSpace
        } else {
            val startToken = tokens[0]
            val endToken = tokens.last()
            startEnd = Pair(startToken.tokenRange.start, endToken.tokenRange.end)
            assert(startToken.position in corrections && endToken.position in corrections)
            val isDelete = ArrayList<Boolean>()

            for (token in tokens) {
                if (corrections.containsKey(token.position) &&
                    corrections.get(token.position)!!.tag == TagDelete.value) {
                    isDelete.add(true)
                } else {
                    isDelete.add(false)
                }
            }
            tag = if (isDelete.all { it }) TagDelete.value else TagKeep.value
            word = endToken.text
            withSpace = endToken.tokenRange.withSpace
        }

        if (tag == TagDelete.value) {
            startEnd = Pair(startEnd.first, startEnd.second + 1)
        }

        if (tag == TagDelete.value && !withSpace && word != " ") {
            return Pair(startEnd.first, startEnd.second + 1)
        } else {
            return Pair(startEnd.first, startEnd.second)
        }
    }

    private fun createUnderlineRange(tokens: List<Token>): Pair<Int, Int> {
        return if (tokens.size == 1) {
            Pair(tokens[0].tokenRange.start, tokens[0].tokenRange.end)
        } else {
            Pair(tokens[0].tokenRange.start, tokens.last().tokenRange.end)
        }
    }

    private fun createMessage(tokens: List<Token>): String {
        if (tokens.size == 1) {
            assert(tokens[0].position in corrections)
            return corrections[tokens[0].position]!!.errorClass
        } else {
            return "Complex error"
        }
    }

    private fun createReplacement(tokens: List<Token>): String {
        var replacement = ""
        for (token in tokens) {
            var tokenReplacement = ""
            if (token.position in corrections) {
                for (cToken in parseTokenCorrection(token)) {
                    tokenReplacement = updateReplacementString(tokenReplacement, cToken.text, withSpace = cToken.tokenRange.withSpace)
                }
            } else {
                tokenReplacement = token.text
            }
            replacement = updateReplacementString(replacement, tokenReplacement, withSpace = token.tokenRange.withSpace)
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
