package io.kinference.algorithms.completion.generation.matcher

import info.debatty.java.stringsimilarity.Levenshtein

abstract class PrefixMatcher {
    fun prefixTokens(prefix: String): IntArray {
        return prefixTokensByErr(prefix, errLimit = 0)[1]
    }

    fun notPrefixTokens(prefix: String): IntArray {
        return prefixTokensByErr(prefix, errLimit = 0)[0]
    }

    abstract fun prefixTokensByErr(prefix: String, errLimit: Int = 0): Array<IntArray>

    companion object {
        private val levenshtein = Levenshtein()

        fun levenshtein(s1: String, s2: String) = levenshtein.distance(s1, s2).toInt()

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

    }
}

