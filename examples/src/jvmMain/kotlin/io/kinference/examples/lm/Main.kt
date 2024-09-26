package io.kinference.examples.lm

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import io.kinference.core.KIEngine
import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXDataType
import io.kinference.examples.downloadFile
import io.kinference.examples.resourcesPath
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.arrays.NDArrayCore
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.ort.ORTEngine
import io.kinference.utils.CommonDataLoader
import io.kinference.utils.PredictionConfigs
import io.kinference.utils.inlines.InlineInt
import io.kinference.utils.toIntArray
import okio.Path.Companion.toPath

// Softmax function
fun softmax(logits: FloatArray): FloatArray {
    val maxLogit = logits.maxOrNull() ?: 0.0f
    val expLogits = logits.map { Math.exp((it - maxLogit).toDouble()).toFloat() }.toFloatArray()
    val sumExp = expLogits.sum()
    return expLogits.map { it / sumExp }.toFloatArray()  // Normalize
}

// Top-K function to get the top K probabilities and their indices
fun topK(probs: FloatArray, k: Int): Pair<FloatArray, IntArray> {
    val indexedProbs = probs.mapIndexed { index, prob -> index to prob }
    val sortedProbs = indexedProbs.sortedByDescending { it.second }.take(k)
    val topProbs = sortedProbs.map { it.second }.toFloatArray()
    val topIndices = sortedProbs.map { it.first }.toIntArray()
    return Pair(topProbs, topIndices)
}

fun transformToFloatArray2D(original: FloatArray, n: Int): Array<FloatArray> {
    // Calculate how many sub-arrays (rows) we will have
    val rowCount = original.size / n

    // Create a new 2D array to store the result
    val result = Array(rowCount) { FloatArray(n) }

    // Fill the new 2D array with sub-arrays from the original array
    for (i in 0 until rowCount) {
        // Copy the next n elements into the current row
        result[i] = original.sliceArray(i * n until (i + 1) * n)
    }

    return result
}

suspend fun mainONNXRuntimeValidation() {
    val modelBytes = CommonDataLoader.bytes("$resourcesPath/gpt2-lm-head-10.onnx".toPath())
    val model = ORTEngine.loadModel(modelBytes)

    val inputTestTensor = ORTEngine.loadData("$resourcesPath/test_data_set_0/input_0.pb".toPath(), ONNXDataType.ONNX_TENSOR)
    val realOutput = model.predict(listOf(inputTestTensor))
    println(realOutput)
    val output = realOutput["output1"]!!.data as OnnxTensor
    val logits = output.value as Array<Array<Array<FloatArray>>>
    val lastTokenLogits = logits[0][0][7] // shape: [50257]
    val lastTokenProbs = softmax(lastTokenLogits)
    val topK = topK(lastTokenProbs, 5)
    val topKIndices = topK.second
    println(topKIndices.joinToString(", "))
}

suspend fun mainKIValidation() {
    val modelBytes = CommonDataLoader.bytes("$resourcesPath/gpt2-lm-head-10.onnx".toPath())
    val model = KIEngine.loadModel(modelBytes, optimize = true, predictionConfig = PredictionConfigs.NoAllocator)

    val tokenizer = HuggingFaceTokenizer.newInstance("gpt2")

    val inputTestTensor = KIEngine.loadData("$resourcesPath/test_data_set_0/input_0.pb".toPath(), ONNXDataType.ONNX_TENSOR)
    val realOutput = model.predict(listOf(inputTestTensor))
    println(realOutput)

    val farray = ((realOutput["output1"]!! as KITensor).data as FloatNDArray).array.toArray()
    val farray2d = transformToFloatArray2D(farray, 50257)
    println(farray2d)

    val slicedReal = (realOutput["output1"]!! as KITensor).data.slice(
        starts = intArrayOf(0, 0, 8 - 1, 0),
        ends = intArrayOf(1, 1, 8, 50257),
        steps = intArrayOf(1, 1, 1, 1)
    ) as NumberNDArrayCore
    val fslice = (slicedReal as FloatNDArray).array.toArray()
    println(fslice)
    val softmaxReal = slicedReal.softmax(axis = -1)
    val topKReal = softmaxReal.topK(
        axis = -1,
        k = 5,
        largest = true,
        sorted = true
    )

    val tokenIdReal = (topKReal.second as LongNDArray)[intArrayOf(0,0,0,0)].toInt()
    val decodeReal = tokenizer.decode(longArrayOf(tokenIdReal.toLong()))
    println(decodeReal)
}

// Constants for input and output tensor names used in the GPT-2 model
private const val INPUT_TENSOR_NAME = "input1"
private const val OUTPUT_TENSOR_NAME = "output1" // We use only logits tensor

suspend fun extractTopToken(output: Map<String, KIONNXData<*>>, tokensSize: Int): Long {
    val logits = output[OUTPUT_TENSOR_NAME]!! as KITensor
    val sliced = logits.data.slice(
        starts = intArrayOf(0, 0, tokensSize - 1, 0),   // First batch, first element in the second dimension, last token, first vocab entry
        ends = intArrayOf(1, 1, tokensSize, 50257),     // Same batch, same second dimension, one token step, whole vocab (50257)
        steps = intArrayOf(1, 1, 1, 1)                  // Step of 1 for each dimension
    ) as NumberNDArrayCore
    val softmax = sliced.softmax(axis = -1)
    val topK = softmax.topK(
        axis = -1,                                      // Apply top-k along the last dimension (vocabulary size)
        k = 1,                                          // Retrieve the top 1 element
        largest = true,                                 // We want the largest probabilities (most probable tokens)
        sorted = false                                  // Sorting is unnecessary since we are only retrieving the top 1
    )
    val tokenId = (topK.second as LongNDArray)[intArrayOf(0, 0, 0, 0)]

    return tokenId
}

suspend fun main() {
    val modelUrl = "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx"
    val modelName = "gpt2-lm-head-10"

    println("Downloading model from: $modelUrl")
    downloadFile(modelUrl, "$resourcesPath/$modelName.onnx")

    val modelBytes = CommonDataLoader.bytes("${resourcesPath}/$modelName.onnx".toPath())

    println("Loading model...")
    val model = KIEngine.loadModel(modelBytes, optimize = true, predictionConfig = PredictionConfigs.DefaultAutoAllocator)

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

        outputTokens[idx] = extractTopToken(output, tokensSize + idx)

        val newTokenArray = LongNDArray(1, 1) { _: InlineInt -> outputTokens[idx] }
        currentContext = currentContext.concat(listOf(newTokenArray.unsqueeze(0)), axis = -1)
        print(tokenizer.decode(longArrayOf(outputTokens[idx])))
    }
    println("\n\nDone")
}
