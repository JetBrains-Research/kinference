package io.kinference.examples.classification

import io.kinference.examples.downloadFile
import io.kinference.examples.resourcesPath
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.FloatNDArray.Companion.invoke
import io.kinference.ort.ORTEngine
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.CommonDataLoader
import io.kinference.utils.inlines.InlineInt
import io.kinference.utils.toLongArray
import okio.Path.Companion.toPath
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.embedded.dogsCatsSmallDatasetPath
import org.jetbrains.kotlinx.dl.dataset.generator.FromFolders
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.impl.preprocessing.*
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import java.awt.image.BufferedImage
import java.io.File
import kotlin.collections.mutableMapOf

// Constants for input and output tensor names used in the CaffeNet model
private const val INPUT_TENSOR_NAME = "data_0"
private const val OUTPUT_TENSOR_NAME = "prob_1"

// Preprocessing pipeline for input images using KotlinDL
private val preprocessing = pipeline<BufferedImage>()
    .resize {
        outputWidth = 224
        outputHeight = 224
        interpolation = InterpolationType.BILINEAR
    }
    .convert { colorMode = ColorMode.BGR }
    .toFloatArray { }
    .call(InputType.CAFFE.preprocessing())

// Path to the small dataset of dogs vs cats images (100 images)
private val dogsVsCatsDatasetPath = dogsCatsSmallDatasetPath()

/**
 * Creates a Map of input tensors categorized by their respective classes (e.g., "cat" and "dog").
 *
 * This function reads images from the dataset, preprocesses them,
 * transposes the tensors to the required format, and groups them
 * based on their class label.
 *
 * @return A Map where the keys are the class labels (e.g., "cat" and "dog"),
 * and the values are lists of KITensor objects representing the input tensors
 * for each class.
 */
private suspend fun createInputs(): Map<String, List<ORTTensor>> {
    val dataset = OnFlyImageDataset.create(
        File(dogsVsCatsDatasetPath),
        FromFolders(mapping = mapOf("cat" to 0, "dog" to 1)),
        preprocessing
    ).shuffle()


    val tensorShape = intArrayOf(1, 224, 224, 3)        // Original tensor shape is [batch, width, height, channel]
    val permuteAxis = intArrayOf(0, 3, 1, 2)            // Permutations for shape [batch, channel, width, height]
    val inputTensors = mutableMapOf<String, MutableList<ORTTensor>>()

    for (i in 0 until dataset.xSize()) {
        val inputData = dataset.getX(i)
        val inputClass = if (dataset.getY(i).toInt() == 0) "cat" else "dog"
        val floatNDArray = FloatNDArray(tensorShape) { index: InlineInt -> inputData[index.value] }.transpose(permuteAxis)  // Create an NDArray from the image data
        val inputTensor = ORTTensor(floatNDArray.array.toArray(), floatNDArray.shape.toLongArray(), INPUT_TENSOR_NAME)      // Transpose and create a tensor from the NDArray
        inputTensors.putIfAbsent(inputClass, mutableListOf())
        inputTensors[inputClass]!!.add(inputTensor)
    }

    return inputTensors
}

/**
 * Displays the top 5 predictions with their corresponding labels and scores.
 *
 * @param predictions The predicted scores in a multidimensional array format.
 * @param classLabels The list of class labels corresponding to the predictions.
 * @param originalClass The actual class label of the instance being predicted.
 */
private fun displayTopPredictions(predictions: ORTTensor, classLabels: List<String>, originalClass: String) {
    val predictionArray = predictions.toFloatArray()
    val indexedScores = predictionArray.withIndex().sortedByDescending { it.value }.take(5)

    println("\nOriginal class: $originalClass")
    println("Top 5 predictions:")
    for ((index, score) in indexedScores) {
        val predictedClassLabel = if (index in classLabels.indices) classLabels[index] else "Unknown"
        println("${predictedClassLabel}: ${"%.2f".format(score * 100)}%")
    }
}

suspend fun main() {
    val modelUrl = "https://github.com/onnx/models/raw/main/validated/vision/classification/caffenet/model/caffenet-12.onnx"
    val synsetUrl = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    val modelName = "CaffeNet"

    println("Downloading model from: $modelUrl")
    downloadFile(modelUrl, "$resourcesPath/$modelName.onnx")
    println("Downloading synset from: $synsetUrl")
    downloadFile(synsetUrl, "$resourcesPath/synset.txt")

    val modelBytes = CommonDataLoader.bytes("$resourcesPath/$modelName.onnx".toPath())
    val classLabels = File("$resourcesPath/synset.txt").readLines()

    println("Loading model...")
    val model = ORTEngine.loadModel(modelBytes)
    println("Creating inputs...")
    val inputTensors = createInputs()

    println("Starting inference...")
    inputTensors.forEach { dataClass ->
        dataClass.value.forEach { tensor ->
            val actualOutputs = model.predict(listOf(tensor))
            val predictions = actualOutputs[OUTPUT_TENSOR_NAME]!! as ORTTensor
            displayTopPredictions(predictions, classLabels, dataClass.key)
        }
    }
}
