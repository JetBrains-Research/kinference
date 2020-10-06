package io.kinference.completion

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class BPETokenizerTest {
    companion object {
        const val vocabPath = "/Users/aleksandr.khvorov/.cache/torch/transformers/71cc2431cf0b5bbe7a23601a808ed322c90251c8261b46f04970140a3c2c1cb4.1512018be4ba4e8726e41b9145129dc30651ea4fec86aa61f4b9f40bf94eac71"
        const val mergesPath = "/Users/aleksandr.khvorov/.cache/torch/transformers/4faf7afb02a1ea7d2944e9ba7a175c7b8de4957cdbae75cd5ddffc7c7643ebbc.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda"
    }

    @Test
    @Tag("heavy")
    fun testEncodeSmallExample() {
        val tokenizer = BPETokenizer(vocabPath, mergesPath)
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
