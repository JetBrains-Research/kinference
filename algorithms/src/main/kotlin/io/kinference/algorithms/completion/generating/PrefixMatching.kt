package io.kinference.algorithms.completion.generating

import io.kinference.algorithms.completion.BPETokenizer
import kotlin.math.max
import kotlin.math.min

abstract class PrefixMatcher {
    fun prefixTokens(prefix: String): IntArray {
        return prefixTokensByErr(prefix, errLimit = 0)[1]
    }

    fun notPrefixTokens(prefix: String): IntArray {
        return prefixTokensByErr(prefix, errLimit = 0)[0]
    }

    abstract fun prefixTokensByErr(prefix: String, errLimit: Int = 0): Array<IntArray>

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

            val matrix = Array(s2.length + 1) { IntArray(s1.length + 1) }
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

            var minValue = Int.MAX_VALUE
            for (row in matrix) minValue = min(row[row.size - 1], minValue)

            return min(minValue, currColumn.min() ?: Int.MAX_VALUE)
        }
    }
}

class FuzzyPrefixMatcher(val tokenizer: BPETokenizer) : PrefixMatcher() {
    private val tokens: List<String>
    private val origInds: IntArray
    private val trie: Trie

    data class MatchedSubtrie(val start: Int, val finish: Int, var errorsCount: Int)

    inner class Trie {
        var start = tokenizer.vocabSize
        var finish = 0
        val dict = HashMap<Char, Trie>()

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

        fun prefixInds(word: String, errLimit: Int = 0): List<MatchedSubtrie> {
            if (word.isEmpty() || dict.isEmpty()) {
                if (start == 47623) {
                    val a = 0
                }
                return listOf(MatchedSubtrie(start, finish + 1, 0))
            }

            if (!dict.containsKey(word[0])) {
                var minWithSuffix = finish
                for (node in dict.values) {
                    minWithSuffix = min(minWithSuffix, node.start)
                }
                if (start == 47623) {
                    val a = 0
                }
                return listOf(MatchedSubtrie(start, minWithSuffix, 0))
            }

            if (word[0] == ' ') {
                return dict[word[0]]!!.prefixInds(word.substring(1), errLimit)
            }

            val result: MutableList<MatchedSubtrie> = ArrayList()

            if (errLimit > 0) {
                for (symbol in dict.keys) {
                    if (symbol == word[0]) {
                        continue
                    } else if (symbol.isLetter()) {
                        result.addAll(dict[symbol]!!.prefixInds(word.substring(1), errLimit - 1))  // replace
                        result.addAll(dict[symbol]!!.prefixInds(word, errLimit - 1))  // insert
                    }
                }

                result.addAll(prefixInds(word.substring(1), errLimit - 1))  // delete
                result.onEach { it.errorsCount++ }
            }

            result.addAll(dict[word[0]]!!.prefixInds(word.substring(1), errLimit))  // correct

            return result
        }
    }

    init {
        val tokensInds = Array(tokenizer.vocabSize) { Pair(it, tokenizer.decode(it)) }.sortedBy { it.second }
//        val allTokens = (0 until tokenizer.vocabSize)
//            .map { tokenizer.decode(it) }
//        val tokensInds = allTokens.mapIndexed { index, s -> Pair(index, s) }
//            .sortedBy { it.second }
        origInds = IntArray(tokensInds.size) { tokensInds[it].first }
        tokens = tokensInds.map { it.second }

        trie = Trie()
        for (i in tokens.indices) {
            trie.add(tokens[i], i)
        }
    }

//    private fun trieToMap(trie: Trie): Map<String, Any> {
//        return trie.dict.map { entry -> Pair("" + entry.key, trieToMap(entry.value)) }.toMap()
//    }

//    fun dumpTrie() {
//        val obj = JsonObject(trieToMap(trie))
//
//    }

    //    @lru_cache(maxsize = 50)
    override fun prefixTokensByErr(prefix: String, errLimit: Int): Array<IntArray> {
        if (errLimit < 0) {
            return arrayOf(origInds)
        }

        val edges = trie.prefixInds(prefix, errLimit).sortedBy { it.start * tokenizer.vocabSize + it.finish }  //  * tokenizer.vocabSize * tokenizer.vocabSize + it.second * tokenizer.vocabSize + it.third
        //val bad = edges.filterIndexed { index, triple -> index < edges.size - 1 && triple.second > edges[index + 1].first || index > 0 && triple.first < edges[index - 1].second }

        var prevStart = 0
        val result = Array<MutableList<Int>>(errLimit + 2) { ArrayList() }

        for (subtrie in edges) {
            // TODO: it's tokenizer bug, arguments should be (prevStart, triple.first)
//            if (prevStart < start) {
            result[0].addAll(origInds.slice(prevStart until subtrie.start))
//            }
            prevStart = subtrie.finish
            result[subtrie.errorsCount + 1].addAll(origInds.slice(subtrie.start until subtrie.finish))
        }

        result[0].addAll(origInds.slice(prevStart until origInds.size))

        return Array(result.size) { result[it].toIntArray() }
    }
}
