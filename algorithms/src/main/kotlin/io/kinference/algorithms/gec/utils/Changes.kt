package io.kinference.algorithms.gec.utils

import io.kinference.algorithms.gec.preprocessing.*
import java.lang.IllegalArgumentException

internal object EmptyTags {
    private val tags: List<Tag> = listOf(tagPad, tagUnk, tagKeep)
    private val values = tags.map { it.value }

    fun isEmptyTag(tag: String): Boolean{
        return values.contains(tag)
    }
}

internal object AppendNoSpace{
    private val values: Set<String> = setOf("-", ",", ".", ";", ":", "!", "?", "'s", "'m", "'d", "'ve", "'ll", "'re", "n't", "'t")

    fun inNoSpaceCharacters(tok: String): Boolean{
        return values.contains(tok)
    }
}

data class TokenChanges(val replacement: String,
                        var tokenizedReplacement: List<String>? = null,
                        val usedTokensNum: Int = 1){

    init {
        if (tokenizedReplacement == null){
            tokenizedReplacement = listOf(replacement)
        }
    }
}

class TokenChangesGenerator(val tokens: List<Token>, val tags: List<String>, val verbsFormVocabulary: VerbsFormVocabulary){
    fun generateTokenChanges(): List<TokenChanges?>{
        val changesList = ArrayList<TokenChanges?>()
        var idx = 0
        while (idx < tokens.size){
            if (EmptyTags.isEmptyTag(tags[idx])){
                changesList.add(null)
                idx += 1
                continue
            }
            val pairTokenChange = applyTokenTag(idx)
            idx = pairTokenChange.first
            changesList.add(pairTokenChange.second!!)

            for (index in 0 until pairTokenChange.second!!.usedTokensNum - 1){
                changesList.add(TokenChanges(replacement = "", tokenizedReplacement = emptyList(), usedTokensNum = 1))
            }
            idx += pairTokenChange.second!!.usedTokensNum
        }
        return changesList
    }

    private fun applyTokenTag(idx: Int): Pair<Int, TokenChanges?>{
        if (EmptyTags.isEmptyTag(tags[idx])){
            throw IllegalArgumentException("tag_str should be a meaningful error token not ${tags[idx]}")
        }
        val pairDelete = applyDeleteTag(idx)
        val pairReplace = applyReplaceTag(idx)
        val pairTransform = applyTransformTag(idx)
        val pairAppend = applyAppendTag(idx)
        val pairMerge = applyMergeTag(idx)

        if (pairDelete.second != null){
            return pairDelete
        }
        else if (pairReplace.second != null){
            return pairReplace
        }

        else if (pairTransform.second != null){
            return pairTransform
        }
        else if (pairAppend.second != null){
            return pairAppend
        }
        else if (pairMerge.second != null){
            return pairMerge
        }
        else {
            return Pair(idx, TokenChanges(tokens[idx].text))
        }
    }

    private fun applyDeleteTag(idx: Int): Pair<Int, TokenChanges?>{
        if (tags[idx] != "\$DELETE"){
            return Pair(idx, null)
        }
        return Pair(idx, TokenChanges(replacement = "", tokenizedReplacement = emptyList()))
    }

    private fun applyReplaceTag(idx: Int): Pair<Int, TokenChanges?>{
        if (!tags[idx].startsWith("\$REPLACE_")){
            return Pair(idx, null)
        }
        val replaceWord = tags[idx].split("_", limit = 2).last()
        return Pair(idx, TokenChanges(replacement = replaceWord))
    }

    private fun applyTransformTag(idx: Int): Pair<Int, TokenChanges?>{
        if (!tags[idx].startsWith("\$TRANSFORM_")){
            return Pair(idx, null)
        }
        val tokenText = tokens[idx].text
        val tag = tags[idx]

        if(tag.startsWith("\$TRANSFORM_CASE")){
            val splits = tag.split("_", limit = 3)
            return Pair(idx, TokenChanges(transformUsingCase(tokenText, case= splits.last())))
        }
        if (tag.startsWith("\$TRANSFORM_VERB")){
            val splits = tag.split("_", limit = 3)
            return Pair(idx, TokenChanges(transformUsingVerb(tokenText,
                form = splits.last(), verbsVocab = verbsFormVocabulary)))
        }
        if (tag.startsWith("\$TRANSFORM_SPLIT")){
            val parts = transformUsingSplit(tokenText)
            return Pair(idx, TokenChanges(replacement=parts.joinToString { " " }, tokenizedReplacement = parts))
        }
//        if (tag.startsWith("\$TRANSFORM_AGREEMENT")){
//            val splits = tag.split("_", limit = 2)
//            return TokenChanges((, tokenText, form=splits.last()))
//        }
        throw IllegalArgumentException("Unknown transformation type $tag")
    }

    private fun applyAppendTag(idx: Int): Pair<Int, TokenChanges?>{
        if(!tags[idx].startsWith("\$APPEND_")){
            return Pair(idx, null)
        }

        val tokenText = tokens[idx].text
        val appendWord = tags[idx].split("_", limit = 2).last()

        if (tags[idx] != "\$APPEND_-"){
            return if (tokens[idx].isStartToken()){
                Pair(idx, TokenChanges("$appendWord "))
            } else if (AppendNoSpace.inNoSpaceCharacters(appendWord)){
                Pair(idx, TokenChanges("${tokenText}${appendWord}", tokenizedReplacement= listOf(tokenText, appendWord)))
            } else Pair(idx, TokenChanges("$tokenText $appendWord", tokenizedReplacement= listOf(tokenText, appendWord)))
        }

        //APPEND_-
        var replacement = tokenText
        val tokenizedReplacement = mutableListOf(tokenText)

        var newIdx = idx
        val startIdx = idx
        while (newIdx < tags.size - 1 && tags[newIdx] == "\$APPEND_-"){
            replacement = replacement + "-" + tokens[newIdx+1].text
            tokenizedReplacement.add("-")
            tokenizedReplacement.add(tokens[newIdx + 1].text)
            newIdx += 1
        }
        return Pair(newIdx, TokenChanges(replacement, tokenizedReplacement, usedTokensNum = newIdx - startIdx + 1))
    }

    private fun applyMergeTag(idx: Int): Pair<Int, TokenChanges?>{

        if (!tags[idx].startsWith("\$MERGE_")){
            return Pair(idx, null)
        }

        var replacement = tokens[idx].text
        val tokenizedReplacement = mutableListOf(tokens[idx].text)

        var newIdx = idx
        val startIdx = idx
        while (newIdx < tags.size - 1 && (tags[newIdx] == "\$MERGE_SPACE" || tags[newIdx] == "\$MERGE_HYPHEN")){
            val mergeSymbol =
                if (tags[newIdx] == "\$MERGE_SPACE")
                     " "
                else "-"

            replacement = "${replacement}${mergeSymbol}${tokens[newIdx + 1].text}"
            if (mergeSymbol == "-"){
                tokenizedReplacement.add(mergeSymbol)
            }
            tokenizedReplacement.add(tokens[newIdx + 1].text)
            newIdx += 1
        }
        return Pair(newIdx, TokenChanges(replacement, tokenizedReplacement, usedTokensNum=newIdx - startIdx + 1))
    }
}
