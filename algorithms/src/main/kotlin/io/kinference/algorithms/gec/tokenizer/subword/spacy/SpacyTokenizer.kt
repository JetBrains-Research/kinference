package io.kinference.algorithms.gec.tokenizer.subword.spacy

import io.kinference.algorithms.gec.tokenizer.Tokenizer
import io.kinference.algorithms.gec.tokenizer.subword.spacy.en.SpacyEnglish
import io.kinference.algorithms.gec.tokenizer.subword.spacy.en.Vocab

/**
 * Tokenizer class implements spacy tokenization algorithm based on this article https://spacy.io/usage/linguistic-features#how-tokenizer-works
 * @param prefix prefix class for prefix matching
 * @param infix infix cass for infix matching
 * @param suffix suffix class for suffix matching
 * @param specialCases for right tokenization for complex variants
 * @param vocab literally for producing tokens but for now do nothing
 */
class SpacyTokenizer(
    val prefix: Regex,
    private val infix: Regex,
    private val suffix: Regex,
    val specialCases: SpacyTokenizerSpecialCases,
    private val vocab: Vocab
) : Tokenizer {
    companion object {

        fun loadEnglish(): SpacyTokenizer {
            return SpacyTokenizer(SpacyEnglish.Prefix.regex, SpacyEnglish.Infix.regex, SpacyEnglish.Suffix.regex, SpacyTokenizerSpecialCases(), Vocab())
        }
    }

    private class TokenSplits {

        val prefixes = ArrayList<String>()
        val suffixes = ArrayList<String>()
        var word: String? = null
        val wordTokens = ArrayList<String>()
        var isSpecial = false

        fun toList(): List<String> {
            return prefixes + wordTokens + suffixes.reversed()
        }
    }


    override fun tokenize(text: String): List<String> {
        var i = 0
        var start = 0
        val result = ArrayList<String>()

        var inWS = text[0].isWhitespace()

        for (uc in text) {
            if (uc.isWhitespace() != inWS) {
                if (start < i) {
                    val span = text.substring(startIndex = start, endIndex = i)
                    result += tokenizeToken(span)
                }

                start = if (uc == ' ') {
                    i + 1
                } else {
                    i
                }
                inWS = !inWS
            }
            i += 1
        }
        if (start < i) {
            val span = text.substring(startIndex = start)
            result += tokenizeToken(span)
        }
        return result
    }

    private fun tokenizeToken(tok: String): List<String> {
        var splits = splitAffixes(tok)
        if (!splits.isSpecial) {
            splits = attachTokens(splits)
        }
        return splits.toList()
    }

    private fun splitAffixes(tok: String): TokenSplits {
        var lastSize = 0
        var tokVar = tok

        val splits = TokenSplits()
        while (tokVar != "" && tokVar.length != lastSize) {
            if (specialCases.get(tokVar) != null) {
                specialCases.get(tokVar)!!.forEach { splits.wordTokens.add(it.orth) }
                splits.isSpecial = true
                break
            }
            lastSize = tokVar.length
            val prefixLength = findPrefix(tokVar)
            var minusPrefix: String? = null
            var prefix: String? = null
            if (prefixLength != 0) {
                prefix = tokVar.substring(startIndex = 0, endIndex = prefixLength)
                minusPrefix = tokVar.substring(startIndex = prefixLength)

                if (minusPrefix != "" && specialCases.get(minusPrefix) != null) {
                    tokVar = minusPrefix
                    splits.prefixes.add(vocab.get(prefix))
                    break
                }
            }

            val suffixLength = findSuffix(tokVar)
            var minusSuffix: String? = null
            var suffix: String? = null
            if (suffixLength != 0) {
                suffix = tokVar.substring(startIndex = tokVar.length - suffixLength)
                minusSuffix = tokVar.substring(startIndex = 0, endIndex = tokVar.length - suffixLength)
                if (minusSuffix != "" && specialCases.get(minusSuffix) != null) {
                    tokVar = minusSuffix
                    splits.suffixes.add(vocab.get(suffix))
                    break
                }
            }
            if (prefixLength != 0 && suffixLength != 0 && (prefixLength + suffixLength) <= tokVar.length) {
                tokVar = tokVar.substring(startIndex = prefixLength, endIndex = tokVar.length - suffixLength)
                splits.prefixes.add(vocab.get(prefix!!))
                splits.suffixes.add(vocab.get(suffix!!))
            } else if (prefixLength != 0) {
                tokVar = minusPrefix!!
                splits.prefixes.add(vocab.get(prefix!!))
            } else if (suffixLength != 0) {
                tokVar = minusSuffix!!
                splits.suffixes.add(vocab.get(suffix!!))
            }
            if (tokVar != "" && specialCases.get(tokVar) != null) {
                break
            }
        }
        splits.word = tokVar
        return splits
    }

    private fun attachTokens(splits: TokenSplits): TokenSplits {
        if (splits.word!!.isNotEmpty()) {
            if (specialCases.urlMatch(splits.word!!)) {
                splits.wordTokens.add(splits.word!!)
            } else {
                val matches = findInfix(splits.word!!)
                var start = 0
                val startBeforeInfixes = start
                for (match in matches) {
                    val startInfix = match.range.first
                    val endInfix = match.range.last + 1

                    if (startInfix == startBeforeInfixes) {
                        continue
                    }

                    if (startInfix != start) {
                        val spanInfix = splits.word!!.substring(startIndex = start, endIndex = startInfix)
                        splits.wordTokens.add(spanInfix)
                    }

                    if (startInfix != endInfix) {
                        val infix = splits.word!!.substring(startIndex = startInfix, endIndex = endInfix)
                        splits.wordTokens.add(infix)
                    }
                    start = endInfix
                }
                val span = splits.word!!.substring(startIndex = start, endIndex = splits.word!!.length)
                if (span != "") {
                    splits.wordTokens.add(span)
                }
            }
        }
        return splits
    }

    private fun findPrefix(tok: String): Int {
        val match = prefix.find(tok)
        return if (match == null) {
            0
        } else {
            match.range.last() + 1 - match.range.first()
        }
    }

    private fun findSuffix(tok: String): Int {
        val match = suffix.find(tok)
        return if (match == null) {
            0
        } else {
            match.range.last() + 1 - match.range.first()
        }
    }

    private fun findInfix(tok: String): Sequence<MatchResult> {
        return infix.findAll(tok)
    }
}
