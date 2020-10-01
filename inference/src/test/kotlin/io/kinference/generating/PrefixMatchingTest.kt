package io.kinference.generating

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

@Tag("heavy")
class PrefixMatchingTests {
    companion object {
        const val vocabPath = "/Users/aleksandr.khvorov/.cache/torch/transformers/71cc2431cf0b5bbe7a23601a808ed322c90251c8261b46f04970140a3c2c1cb4.1512018be4ba4e8726e41b9145129dc30651ea4fec86aa61f4b9f40bf94eac71"
        const val mergesPath = "/Users/aleksandr.khvorov/.cache/torch/transformers/4faf7afb02a1ea7d2944e9ba7a175c7b8de4957cdbae75cd5ddffc7c7643ebbc.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda"
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
        val tokenizer = BPETokenizer(vocabPath, mergesPath)
        val matcher = FuzzyPrefixMatcher(tokenizer, errorLimit = 0)

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
        val tokenizer = BPETokenizer(vocabPath, mergesPath)
        val matcher = FuzzyPrefixMatcher(tokenizer, errorLimit = 1)

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
