package io.kinference.ndarray.generating

import java.lang.Integer.max
import java.lang.Integer.min
import java.util.*
import kotlin.collections.ArrayList

abstract class PrefixMatcher {
    fun prefixTokens(prefix: String): List<Int> {
        return prefixTokensByErr(prefix, err_limit = 0)[1]
    }

    fun notPrefixTokens(prefix: String): List<Int> {
        return prefixTokensByErr(prefix, err_limit = 0)[0]
    }

    abstract fun prefixTokensByErr(prefix: String, err_limit: Int): List<List<Int>>

    companion object {
        fun errorsCount(s1: String, s2: String): Int {
            var cnt = 0
            for (i in s1.indices) {
                if (i >= s2.length) {
                    break
                }
                cnt += if (s1[i] != s2[i]) 1 else 0
            }
            return cnt
        }

        fun levenshtein(s1: String, s2: String): Int {
            if (s1.isEmpty() || s2.isEmpty()) {
                return 0
            }

            val matrix = MutableList(s2.length + 1) { MutableList(s1.length + 1) { 0 } }
            var prevColumn = matrix[0]

            for (i in s1.indices) {
                prevColumn[i + 1] = prevColumn[i] + 1
            }
            var currColumn = matrix[1]

            for (i2 in s2.indices) {
                currColumn[0] = prevColumn[0] + 1

                for (i1 in s1.indices) {
                    if (s1[i1] == s2[i2]) {
                        currColumn[i1 + 1] = prevColumn[i1]
                    } else {
                        val change = 1 + prevColumn[i1]
                        val remove = 1 + prevColumn[i1 + 1]
                        val insert = 1 + currColumn[i1]

                        currColumn[i1 + 1] = min(min(change, remove), insert)
                    }
                }

                if (i2 != s2.length - 1) {
                    prevColumn = currColumn
                    currColumn = matrix[i2 + 2]
                }
            }

            val lastValues = matrix.map { it[it.size - 1] }.toMutableList()
            lastValues.addAll(currColumn)

            return Collections.min(lastValues)
        }
    }
}

class FuzzyPrefixMatcher(val tokenizer: BPETokenizer, errorLimit: Int = 0) : PrefixMatcher() {
    private val tokens: List<String>
    private val origInds: List<Int>
    private val trie: Trie

    inner class Trie {
        var start = tokenizer.vocabSize
        var finish = 0
        private val dict = HashMap<Char, Trie>()

        fun add(word: String, ind: Int) {
            start = min(start, ind)
            finish = max(finish, ind)
            if (word.isEmpty()) {
                return
            }

            if (!dict.containsKey(word[0])) {
                dict[word[0]] = Trie()
            }
            dict[word[0]]!!.add(word.substring(1), ind)
        }

        fun prefixInds(word: String, errLimit: Int = 0): List<Triple<Int, Int, Int>> {
            if (word.isEmpty() || dict.isEmpty()) {
                return listOf(Triple(start, finish + 1, 0))
            }

            if (!dict.containsKey(word[0])) {
                var minWithSuffix = finish
                for (node in dict.values) {
                    minWithSuffix = min(minWithSuffix, node.start)
                }
                return listOf(Triple(start, minWithSuffix, 0))
            }

            if (word[0] == ' ' || errLimit == 0) {
                return dict[word[0]]!!.prefixInds(word.substring(1), errLimit)
            }

            val result = ArrayList<Triple<Int, Int, Int>>()
            for (symbol in dict.keys) {
                if (symbol == word[0]) {
                    continue
                } else if (symbol.isLetter()) {
                    result.addAll(dict[symbol]!!.prefixInds(word.substring(1), errLimit - 1))  // replace
                    result.addAll(dict[symbol]!!.prefixInds(word, errLimit - 1))  // insert
                }
            }

            result.addAll(prefixInds(word.substring(1), errLimit - 1))  // delete

            val notCorrectResults = result.map { Triple(it.first, it.second, it.third + 1) }.toMutableList()
            notCorrectResults.addAll(dict[word[0]]!!.prefixInds(word.substring(1), errLimit))  // correct

            return notCorrectResults
        }
    }

    init {
        val tokensInds = (0 until tokenizer.vocabSize)
            .map { tokenizer.decode(it) }
            .mapIndexed { index, s -> Pair(index, s) }
            .sortedBy { it.second }
        origInds = tokensInds.map { it.first }
        tokens = tokensInds.map { it.second }

        trie = Trie()
        for (i in tokens.indices) {
            trie.add(tokens[i], i)
        }
    }

    //    @lru_cache(maxsize = 50)
    override fun prefixTokensByErr(prefix: String, errLimit: Int): List<List<Int>> {
        if (errLimit < 0) {
            return listOf(origInds)
        }

        val edges = trie.prefixInds(prefix, errLimit).sortedBy { it.first * tokenizer.vocabSize * tokenizer.vocabSize + it.second * tokenizer.vocabSize + it.third }

        var prevStart = 0
        val result = MutableList<MutableList<Int>>(errLimit + 2) { ArrayList() }

        for (triple in edges) {
            val start = triple.first
            val finish = triple.second
            val errCount = triple.third

            result[0].addAll(origInds.subList(prevStart, triple.first))
            prevStart = triple.second
            result[errCount + 1].addAll(origInds.subList(start, finish))
        }

        result[0].addAll(origInds.subList(prevStart, origInds.size))

        return result
    }
}
