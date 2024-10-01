package io.kinference.examples.lm

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.examples.downloadFile
import io.kinference.examples.extractTopToken
import io.kinference.examples.cacheDirectory
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.FloatNDArray.Companion.invoke
import io.kinference.ort.ORTData
import io.kinference.ort.ORTEngine
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.CommonDataLoader
import io.kinference.utils.inlines.InlineInt
import io.kinference.utils.toIntArray
import okio.Path.Companion.toPath

// Constants for input and output tensor names used in the GPT-2 model
private const val INPUT_TENSOR_NAME = "input1"
private const val OUTPUT_TENSOR_NAME = "output1" // We use only logits tensor

suspend fun main() {
    val modelUrl = "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx"
    val modelName = "gpt2-lm-head-10"

    println("Downloading model from: $modelUrl")
    downloadFile(modelUrl, "$modelName.onnx") //GPT-2 from model zoo is around 650 Mb, adjust your timeout if needed

    println("Loading model...")
    val model = ORTEngine.loadModel("$cacheDirectory/$modelName.onnx".toPath())

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

    val input = ORTTensor(tokens, longArrayOf(1, 1, tokensSize.toLong()))
    var currentContext = input.clone(INPUT_TENSOR_NAME)

    print("Here goes the test text for generation:\n$testString")

    for (idx in 0 until predictionLength) {
        val inputTensor = listOf(currentContext)
        val output = model.predict(inputTensor)

        outputTokens[idx] = extractTopToken(convertToKITensorMap(output), tokensSize + idx, OUTPUT_TENSOR_NAME)

        val newTokenArray = tokens + outputTokens.slice(IntRange(0, idx))
        currentContext = ORTTensor(newTokenArray, longArrayOf(1, 1, tokensSize + idx + 1L), INPUT_TENSOR_NAME)
        print(tokenizer.decode(longArrayOf(outputTokens[idx])))
    }
    println("\n\nDone")
}

private suspend fun convertToKITensorMap(outputs: Map<String, ORTData<*>>): Map<String, KITensor> {
    return outputs.map { (name, ortTensor) ->
        val ortTensor = ortTensor as ORTTensor
        val data = ortTensor.toFloatArray()
        val shape = ortTensor.shape.toIntArray()
        val ndArray = FloatNDArray(shape) { idx: InlineInt -> data[idx.value] }
        val kiTensor = ndArray.asTensor(name)
        return@map name to kiTensor
    }.toMap()
}
