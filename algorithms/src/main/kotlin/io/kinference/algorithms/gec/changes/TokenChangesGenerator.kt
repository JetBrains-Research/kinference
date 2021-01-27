package io.kinference.algorithms.gec.changes

import io.kinference.algorithms.gec.GECTag
import io.kinference.algorithms.gec.corrector.correction.SentenceCorrections
import io.kinference.algorithms.gec.preprocessing.VerbsFormVocabulary

/**
 * class which generates TokenChages on every iteration
 */
class TokenChangesGenerator(val tokens: List<SentenceCorrections.GECToken>, private val tags: List<String>, val verbsFormVocabulary: VerbsFormVocabulary) {
    fun generateTokenChanges(): List<TokenChanges?> {
        val changesList = ArrayList<TokenChanges?>()
        var idx = 0
        while (idx < tokens.size) {
            if (GECTag.from(tags[idx])?.isNonChanging == true) {
                changesList.add(null)
                idx += 1
                continue
            }
            val pairTokenChange = applyTokenTag(idx)
            idx = pairTokenChange.first
            changesList.add(pairTokenChange.second!!)

            for (index in 0 until pairTokenChange.second!!.usedTokensNum - 1) {
                changesList.add(TokenChanges(replacement = "", tokenizedReplacement = emptyList(), usedTokensNum = 1))
            }
            idx += pairTokenChange.second!!.usedTokensNum
        }
        return changesList
    }

    private fun applyTokenTag(idx: Int): Pair<Int, TokenChanges?> {
        require(GECTag.from(tags[idx])?.isNonChanging?.not() ?: true ) {
            throw IllegalArgumentException("tag_str should be a meaningful error token not ${tags[idx]}")
        }
        val pairDelete = applyDeleteTag(idx)
        val pairReplace = applyReplaceTag(idx)
        val pairTransform = applyTransformTag(idx)
        val pairAppend = applyAppendTag(idx)
        val pairMerge = applyMergeTag(idx)

        return when {
            pairDelete.second != null -> {
                pairDelete
            }
            pairReplace.second != null -> {
                pairReplace
            }
            pairTransform.second != null -> {
                pairTransform
            }
            pairAppend.second != null -> {
                pairAppend
            }
            pairMerge.second != null -> {
                pairMerge
            }
            else -> {
                Pair(idx, TokenChanges(tokens[idx].text))
            }
        }
    }

    private fun applyDeleteTag(idx: Int): Pair<Int, TokenChanges?> {
        if (tags[idx] != "\$DELETE") {
            return Pair(idx, null)
        }
        return Pair(idx, TokenChanges(replacement = "", tokenizedReplacement = emptyList()))
    }

    private fun applyReplaceTag(idx: Int): Pair<Int, TokenChanges?> {
        if (!tags[idx].startsWith("\$REPLACE_")) {
            return Pair(idx, null)
        }
        val replaceWord = tags[idx].split("_", limit = 2).last()
        return Pair(idx, TokenChanges(replacement = replaceWord))
    }

    private fun applyTransformTag(idx: Int): Pair<Int, TokenChanges?> {
        if (!tags[idx].startsWith("\$TRANSFORM_")) {
            return Pair(idx, null)
        }
        val tokenText = tokens[idx].text
        val tag = tags[idx]

        if (tag.startsWith("\$TRANSFORM_CASE")) {
            val splits = tag.split("_", limit = 3)
            return Pair(idx, TokenChanges(Transformations.transformUsingCase(tokenText, case = splits.last())))
        }
        if (tag.startsWith("\$TRANSFORM_VERB")) {
            val splits = tag.split("_", limit = 3)
            return Pair(
                idx, TokenChanges(
                    Transformations.transformUsingVerb(
                        tokenText,
                        form = splits.last(), verbsVocab = verbsFormVocabulary
                    )
                )
            )
        }
        if (tag.startsWith("\$TRANSFORM_SPLIT")) {
            val parts = Transformations.transformUsingSplit(tokenText)
            return Pair(idx, TokenChanges(replacement = parts.joinToString { " " }, tokenizedReplacement = parts))
        }
        if (tag.startsWith("\$TRANSFORM_AGREEMENT")){
            val form = tag.split("_", limit = 2).last()
            val newToken = Transformations.transformUsingPlural(token = tokenText, form = form)
            return Pair(idx, TokenChanges(replacement = newToken))
        }
        error("Unknown transformation type $tag")
    }

    private fun applyAppendTag(idx: Int): Pair<Int, TokenChanges?> {
        if (!tags[idx].startsWith("\$APPEND_")) {
            return Pair(idx, null)
        }

        val tokenText = tokens[idx].text
        val appendWord = tags[idx].split("_", limit = 2).last()

        if (tags[idx] != "\$APPEND_-") {
            return when {
                tokens[idx].isStartToken() -> {
                    Pair(idx, TokenChanges("$appendWord "))
                }
                AppendNoSpace.inNoSpaceCharacters(appendWord) -> {
                    Pair(idx, TokenChanges("${tokenText}${appendWord}", tokenizedReplacement = listOf(tokenText, appendWord)))
                }
                else -> Pair(idx, TokenChanges("$tokenText $appendWord", tokenizedReplacement = listOf(tokenText, appendWord)))
            }
        }

        //APPEND_-
        var replacement = tokenText
        val tokenizedReplacement = mutableListOf(tokenText)

        var newIdx = idx
        val startIdx = idx
        while (newIdx < tags.size - 1 && tags[newIdx] == "\$APPEND_-") {
            replacement = replacement + "-" + tokens[newIdx + 1].text
            tokenizedReplacement.add("-")
            tokenizedReplacement.add(tokens[newIdx + 1].text)
            newIdx += 1
        }
        return Pair(newIdx, TokenChanges(replacement, tokenizedReplacement, usedTokensNum = newIdx - startIdx + 1))
    }

    private fun applyMergeTag(idx: Int): Pair<Int, TokenChanges?> {
        if (!tags[idx].startsWith("\$MERGE_")) {
            return Pair(idx, null)
        }

        var replacement = tokens[idx].text
        val tokenizedReplacement = mutableListOf(tokens[idx].text)

        var newIdx = idx
        val startIdx = idx
        while (newIdx < tags.size - 1 && (tags[newIdx] == "\$MERGE_SPACE" || tags[newIdx] == "\$MERGE_HYPHEN")) {
            val mergeSymbol = if (tags[newIdx] == "\$MERGE_SPACE") " " else "-"

            replacement = "${replacement}${mergeSymbol}${tokens[newIdx + 1].text}"
            if (mergeSymbol == "-") {
                tokenizedReplacement.add(mergeSymbol)
            }
            tokenizedReplacement.add(tokens[newIdx + 1].text)
            newIdx += 1
        }
        return Pair(newIdx, TokenChanges(replacement, tokenizedReplacement, usedTokensNum = newIdx - startIdx + 1))
    }
}
