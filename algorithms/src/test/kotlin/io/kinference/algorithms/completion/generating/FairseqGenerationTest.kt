package io.kinference.algorithms.completion.generating

import io.kinference.algorithms.completion.*
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class FairseqGenerationTest {
    companion object {
        private val tokenizerConfig = bpeTokenizer
        private val modelConfig = model27
        private val generationConfig = defaultGenerationConfig
    }

//    class MockModelWrapper : ModelWrapper {
//        override fun initLogProbs(inputIds: List<List<Int>>): Pair<List<List<MutableList<Double>>>, List<MutableNDArray>> {
//            assert(inputIds.size == 1)
//            assert(inputIds.size == 1)
//            val probs = listOf(listOf(
//                mutableListOf(0.1, 0.2, 0.1, 0.4, 0.2),
//                mutableListOf(0.4, 0.1, 0.0, 0.3, 0.1)
//            ))
//        }
//
//        override fun getLogProbs(inputIds: List<List<Int>>, past: List<MutableNDArray>): Pair<List<List<MutableList<Double>>>, List<MutableNDArray>> {
//            TODO("Not yet implemented")
//        }
//
//    }

//    @Test
//    @Tag("heavy")
//    fun testMockedModel() {
//        val model = ModelWrapper
//        val tokenizer = BPETokenizer(vocabPath, mergesPath)
//        val generator = FairseqGeneration(model, tokenizer)
//
//        val text = "hello"
//        val prefix = " "
//        val contextIds = tokenizer.encode(text)
//        val result = generator.generate(contextIds, prefix, 5, 3)
//        val variants = result.map { it.second[0].map { pair -> tokenizer.decode(pair.first) } }
//        print(variants)
//    }

    @Test
    @Tag("heavy")
    fun testExecutable() {
        val model = GPT2ModelWrapper(modelConfig)
        val tokenizer = BPETokenizer(tokenizerConfig.vocabPath, tokenizerConfig.mergesPath)
        val generator = FairseqGeneration(model, tokenizer)

        val text = "hello"
        val prefix = " wo"
        val contextIds = tokenizer.encode(text)
        val result = generator.generate(contextIds, prefix, generationConfig)
        val variants = result.map { it[0].map { h -> tokenizer.decode(h.hypothesis) } }

        assertEquals(variants[0].toSet(), setOf(" would", " work", " world", " working", " won"))

//        assertTrue(" would" in variants[0])
//        assertTrue(" world" in variants[0])
//        assertTrue(" won" in variants[0])

        assertTrue(" would be" in variants[1])
        assertTrue(" would you" in variants[1])
        assertTrue(" would not" in variants[1])
        assertTrue(" would have" in variants[1])

        assertTrue(" would be a" in variants[2])
        assertTrue(" would be the" in variants[2])
        assertTrue(" would not be" in variants[2])
        assertTrue(" would have been" in variants[2])
    }
}
