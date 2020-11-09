package io.kinference.runners

import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.loaders.S3Client
import io.kinference.model.Model
import io.kinference.onnx.TensorProto
import io.kinference.utils.Assertions
import io.kinference.utils.DataLoader
import java.io.File
import kotlin.math.pow

object TestRunner {
    private val testData = File("../build/test-data")

    private val delta = (10.0).pow(-3)

    data class TensorTestData(val actual: List<ONNXData>, val expected: List<ONNXData>)

    private fun runTestsFromS3(testPath: String, prefix: String): List<TensorTestData> {
        val toFolder = File(testData, testPath)
        S3Client.copyObjects(prefix, toFolder)
        return runTestsFromFolder(toFolder)
    }

    private fun runTestsFromResources(testPath: String): List<TensorTestData> {
        val path = javaClass.getResource(testPath)!!.path
        return runTestsFromFolder(File(path))
    }

    private fun runTestsFromFolder(path: File): List<TensorTestData> {
        val model = Model.load(File(path, "model.onnx").absolutePath)

        return path.list()!!.filter { "test" in it }.map {
            val inputFiles = File("$path/$it").walk().filter { file -> "input" in file.name }
            val outputFiles = File("$path/$it").walk().filter { file -> "output" in file.name }

            val inputTensors = inputFiles.map { model.graph.prepareInput(TensorProto.ADAPTER.decode(it.readBytes())) }.toList()
            val expectedOutputTensors = outputFiles.map { DataLoader.getTensor(it) }.toList()
            val actualOutputTensors = model.predict(inputTensors)
            TensorTestData(expectedOutputTensors, actualOutputTensors)
        }
    }

    fun runFromS3(path: String, prefix: String, delta: Double = TestRunner.delta) {
        check(runTestsFromS3(path, prefix), delta)
    }

    fun runFromResources(path: String, delta: Double = TestRunner.delta) {
        check(runTestsFromResources(path), delta)
    }

    private fun check(datasets: List<TensorTestData>, delta: Double = TestRunner.delta) {
        for (dataSet in datasets) {
            val (expectedOutputTensors, actualOutputTensors) = dataSet

            val mappedActualOutputTensors = actualOutputTensors.associateBy { it.info.name }

            for (expectedOutputTensor in expectedOutputTensors) {
                val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.info.name] ?: error("Required tensor not found")
                Assertions.assertEquals(expectedOutputTensor as Tensor, actualOutputTensor as Tensor, delta)
            }
        }
    }
}
