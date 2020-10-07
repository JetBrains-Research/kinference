package io.kinference.completion

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class BPETokenizerTest {
    companion object {
        private val tokenizerConfig = bpeTokenizer
    }

    @Test
    @Tag("heavy")
    fun testEncodeSmallExample() {
        val tokenizer = BPETokenizer(tokenizerConfig.vocabPath, tokenizerConfig.mergesPath)
        val text = "1. Modeling Vocabulary for Big Code Machine Learning (https://arxiv.org/pdf/1904.01873.pdf)"
        val targetCodes = listOf(
            16, 13, 9104, 278, 47208, 22528, 329, 4403, 6127, 10850, 18252, 357, 5450, 1378,
            283, 87, 452, 13, 2398, 14, 12315, 14, 1129, 3023, 13, 29159, 4790, 13, 12315, 8
        )
        val codedText = tokenizer.encode(text)
        val decodedText = tokenizer.decode(codedText)
        assertEquals(targetCodes, codedText)
        assertEquals(text, decodedText)
    }
}
