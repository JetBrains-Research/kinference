package io.kinference.completion.generating

import io.kinference.completion.BPETokenizer
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class FairseqGenerationTest {
    companion object {
        const val vocabPath = "/Users/aleksandr.khvorov/.cache/torch/transformers/71cc2431cf0b5bbe7a23601a808ed322c90251c8261b46f04970140a3c2c1cb4.1512018be4ba4e8726e41b9145129dc30651ea4fec86aa61f4b9f40bf94eac71"
        const val mergesPath = "/Users/aleksandr.khvorov/.cache/torch/transformers/4faf7afb02a1ea7d2944e9ba7a175c7b8de4957cdbae75cd5ddffc7c7643ebbc.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda"
        const val baseDir = "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/src"
        const val modelPath = "$baseDir/completion/big/opt/onnxrt/onnx_models/distilgpt2_l3_h12_d256_int8.onnx"
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
        val model = OnnxModelWrapper(modelPath)
        val tokenizer = BPETokenizer(vocabPath, mergesPath)
        val generator = FairseqGeneration(model, tokenizer)

        val text = "hello"
        val prefix = " wo"
        val contextIds = tokenizer.encode(text)
        val result = generator.generate(contextIds, prefix, 5, 3)
        val variants = result.map { it.second[0].map { pair -> tokenizer.decode(pair.first) } }

        assertEquals(variants[0].toSet(), setOf(" would", " work", " world", " working", " won"))

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
