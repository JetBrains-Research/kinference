package io.kinference.algorithms.gec.tokenizer

import io.kinference.algorithms.gec.tokenizer.word.BasicWordTokenizer
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class BasicWordTokenizerTest {
    @Test
    fun `tokenize empty text`() {
        val text = ""
        Assertions.assertEquals(emptyList<String>(), BasicWordTokenizer(toLowerCase = true, shouldNormalizeAccentLetters = true).tokenize(text))
    }

    @Test
    fun `tokenize blank text`() {
        val text = " "
        Assertions.assertEquals(emptyList<String>(), BasicWordTokenizer(toLowerCase = true, shouldNormalizeAccentLetters = true).tokenize(text))
    }

    @Test
    fun `tokenize few words`() {
        val text = "It is a few words."
        Assertions.assertEquals(listOf("it", "is", "a", "few", "words", "."), BasicWordTokenizer(toLowerCase = true, shouldNormalizeAccentLetters = true).tokenize(text))
    }

    @Test
    fun `tokenize few sentences`() {
        val text = "It is a few words. And I would like to add one more. And one more."
        Assertions.assertEquals(
            listOf(
                "it", "is", "a", "few", "words", ".", "and", "i", "would", "like", "to", "add", "one", "more", ".", "and", "one", "more", "."
            ),
            BasicWordTokenizer(toLowerCase = true, shouldNormalizeAccentLetters = true).tokenize(text)
        )
    }


    @Test
    fun `tokenize few sentences with apostrophes`() {
        val text = "It's a few words. And I'll like to add one more. And one more persons' word."
        Assertions.assertEquals(
            listOf(
                "it", "'", "s", "a", "few", "words", ".", "and", "i", "'", "ll", "like", "to", "add", "one", "more", ".", "and", "one", "more", "persons",
                "'", "word", "."
            ),
            BasicWordTokenizer(toLowerCase = true, shouldNormalizeAccentLetters = true).tokenize(text)
        )
    }

    @Test
    fun `tokenize few sentences with accents with stripping`() {
        val text = "It's á few words."
        Assertions.assertEquals(listOf("it", "'", "s", "a", "few", "words", "."), BasicWordTokenizer(toLowerCase = true, shouldNormalizeAccentLetters = true).tokenize(text))
    }

    @Test
    fun `tokenize few sentences with accents without stripping`() {
        val text = "It's á few words."
        Assertions.assertEquals(listOf("it", "'", "s", "á", "few", "words", "."), BasicWordTokenizer(toLowerCase = true, shouldNormalizeAccentLetters = false).tokenize(text))
    }
}
