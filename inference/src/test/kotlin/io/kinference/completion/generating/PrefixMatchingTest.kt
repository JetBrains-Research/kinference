package io.kinference.completion.generating

import io.kinference.completion.BPETokenizer
import io.kinference.completion.bpeTokenizer
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

@Tag("heavy")
class PrefixMatchingTests {
    companion object {
        private val tokenizerConfig = bpeTokenizer
    }

    @Test
    fun testHammingDistance() {
        assert(PrefixMatcher.errorsCount("hello", "hello") == 0)
        assert(PrefixMatcher.errorsCount("hello", "helo") == 1)
        assert(PrefixMatcher.errorsCount("hello", "helllo") == 1)
    }

    @Test
    fun testLevenshteinDistance() {
        assert(PrefixMatcher.levenshtein("hello", "hello") == 0)
        assert(PrefixMatcher.levenshtein("hello", "helo") == 1)
        assert(PrefixMatcher.levenshtein("hello", "helllo") == 1)
        assert(PrefixMatcher.levenshtein("hello", "gelllo") == 2)
        assert(PrefixMatcher.levenshtein("heltelo", "gelllo") == 3)
    }

    @Test
    fun testStrictMatching() {
        val tokenizer = BPETokenizer(tokenizerConfig.vocabPath, tokenizerConfig.mergesPath)
        val matcher = FuzzyPrefixMatcher(tokenizer)

        val prefixes = listOf("sou", "sour", "sor", "he", "", "helloworld")

        prefixes.forEach { prefix ->
            val r = matcher.prefixTokensByErr(prefix, 0)

            assert(r.map { it.size }.sum() == tokenizer.vocabSize)

            for (id in r[0]) {
                val token = tokenizer.decode(id)
                assertFalse(token.startsWith(prefix))  //  || prefix.startsWith(token)
            }

            assert(r.size == 2)
            for (id in r[1]) {
                val token = tokenizer.decode(id)
                assertTrue(token.startsWith(prefix) || prefix.startsWith(token))
            }
        }
    }

//    TODO: make working. For this, fix tokenizer byte symbols
//    @Test
    fun testFuzzyMatching() {
        val tokenizer = BPETokenizer(tokenizerConfig.vocabPath, tokenizerConfig.mergesPath)
        val matcher = FuzzyPrefixMatcher(tokenizer)

        val prefixes = listOf("sor", "sorc", "helo")

        prefixes.forEach { prefix ->
            val r = matcher.prefixTokensByErr(prefix, 1)
            val tokens = r.subList(1, r.size).map { it.map { id -> tokenizer.decode(id) } }

            assert(r.map { it.size }.sum() == tokenizer.vocabSize)
            assert(r.size == 3)

            for (id in r[0]) {
                val token = tokenizer.decode(id)
                assertFalse(token.startsWith(prefix))  //  || prefix.startsWith(token)
            }

            for (id in r[1]) {
                val token = tokenizer.decode(id)
                assertTrue(token.startsWith(prefix) || prefix.startsWith(token))
            }

            for (id in r[2]) {
                val token = tokenizer.decode(id)
                assertEquals(PrefixMatcher.levenshtein(token, prefix), 1)
            }
        }
    }
}
