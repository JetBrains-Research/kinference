package io.kinference.algorithms.completion.bpe

import io.kinference.algorithms.completion.BPETokenizer
import io.kinference.algorithms.completion.CompletionModels
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class BPETokenizerTest {
    @Test
    @Tag("heavy")
    fun testEncodeSmallExample() {
        val (tokenizerConfig, _) = CompletionModels.v4

        val tokenizer = BPETokenizer(tokenizerConfig.vocabPath, tokenizerConfig.mergesPath)
        val text = "1. Modeling Vocabulary for Big Code Machine Learning (https://arxiv.org/pdf/1904.01873.pdf)"
        val targetCodes = intArrayOf(
            16, 13, 9104, 278, 47208, 22528, 329, 4403, 6127, 10850, 18252, 357, 5450, 1378,
            283, 87, 452, 13, 2398, 14, 12315, 14, 1129, 3023, 13, 29159, 4790, 13, 12315, 8
        )
        val codedText = tokenizer.encode(text)
        val decodedText = tokenizer.decode(codedText)
        assertArrayEquals(targetCodes, codedText)
        assertEquals(text, decodedText)
    }
}
