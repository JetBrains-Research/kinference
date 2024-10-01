package io.kinference.examples.lm

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import io.kinference.core.KIEngine
import io.kinference.core.data.tensor.asTensor
import io.kinference.examples.downloadFile
import io.kinference.examples.extractTopToken
import io.kinference.examples.cacheDirectory
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.arrays.NDArrayCore
import io.kinference.utils.CommonDataLoader
import io.kinference.utils.PredictionConfigs
import io.kinference.utils.inlines.InlineInt
import okio.Path.Companion.toPath

// Constants for input and output tensor names used in the GPT-2 model
private const val INPUT_TENSOR_NAME = "input1"
private const val OUTPUT_TENSOR_NAME = "output1" // We use only logits tensor

suspend fun main() {
    val modelUrl = "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx"
    val modelName = "gpt2-lm-head-10"

    println("Downloading model from: $modelUrl")
    downloadFile(modelUrl, "$modelName.onnx")  //GPT-2 from model zoo is around 650 Mb, adjust your timeout if needed

    println("Loading model...")
    val model = KIEngine.loadModel("$cacheDirectory/$modelName.onnx".toPath(), optimize = true, predictionConfig = PredictionConfigs.DefaultAutoAllocator)

    val tokenizer = HuggingFaceTokenizer.newInstance("gpt2", mapOf("modelMaxLength" to "1024"))
    val testString = "Neurogenesis is most active during embryonic development and is responsible for producing " +
        "all the various types of neurons of the organism, but it continues throughout adult life " +
        "in a variety of organisms. Once born, neurons do not divide (see mitosis), and many will " +
        "live the lifespan of the animal, except under extraordinary and usually pathogenic circumstances."
    val encoded = tokenizer.encode(testString)
    val tokens = encoded.ids
    val tokensSize = tokens.size

    val predictionLength = 34
    val outputTokens = LongArray(predictionLength) { 0 }

    val input = LongNDArray(1, tokensSize) { idx: InlineInt -> tokens[idx.value] }.unsqueeze(0)
    var currentContext = input.clone()

    print("Here goes the test text for generation:\n$testString")

    for (idx in 0 until predictionLength) {
        val inputTensor = listOf((currentContext as NDArrayCore).asTensor(INPUT_TENSOR_NAME))
        val output = model.predict(inputTensor)

        outputTokens[idx] = extractTopToken(output, tokensSize + idx, OUTPUT_TENSOR_NAME)

        val newTokenArray = LongNDArray(1, 1) { _: InlineInt -> outputTokens[idx] }
        currentContext = currentContext.concat(listOf(newTokenArray.unsqueeze(0)), axis = -1)
        print(tokenizer.decode(longArrayOf(outputTokens[idx])))
    }
    println("\n\nDone")
}
