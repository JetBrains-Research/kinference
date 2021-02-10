package io.kinference.runners

import io.kinference.data.ONNXData
import io.kinference.model.Model
import io.kinference.onnx.TensorProto
import io.kinference.utils.*
import kotlin.math.pow
import kotlin.test.assertEquals

object AccuracyRunner {
    private val delta = (10.0).pow(-3)

    data class ONNXTestData(val actual: List<ONNXData>, val expected: List<ONNXData>)

    private suspend fun runTestsFromS3(name: String): List<ONNXTestData> {
        val toFolder = name.replace(":", "/")
        return runTestsFromFolder(S3TestDataLoader, toFolder)
    }

    private suspend fun runTestsFromResources(testPath: String): List<ONNXTestData> {
        val path = "build/processedResources/${TestRunner.forPlatform("js", "jvm")}/test/${testPath}"
        return runTestsFromFolder(ResourcesTestDataLoader, path)
    }

    private suspend fun runTestsFromFolder(loader: TestDataLoader, path: String): List<ONNXTestData> {
        val model = Model.load(loader.bytes(TestDataLoader.Path(path, "model.onnx")))
        val files = loader.text(TestDataLoader.Path(path, "descriptor.txt")).lines()
        return files.filter { "test" in it }.groupBy { file -> file.takeWhile { it != '/' } }.map { (group, files) ->
            val inputFiles = files.filter { file -> "input" in file }
            val inputTensorProtos = inputFiles.map { TensorProto.ADAPTER.decode(loader.bytes(TestDataLoader.Path(path, it))) }
            val inputTensors = inputTensorProtos.map{ model.graph.prepareInput(it) }

            val outputFiles =  files.filter { file -> "output" in file }
            val expectedOutputTensors = outputFiles.map { DataLoader.getTensor(loader.bytes(TestDataLoader.Path(path, it))) }.toList()

            val actualOutputTensors = model.predict(inputTensors)
            ONNXTestData(expectedOutputTensors, actualOutputTensors)
        }
    }

    suspend fun runFromS3(name: String, delta: Double = AccuracyRunner.delta) {
        check(runTestsFromS3(name), delta)
    }

    suspend fun runFromResources(path: String, delta: Double = AccuracyRunner.delta) {
        check(runTestsFromResources(path), delta)
    }

    private fun check(datasets: List<ONNXTestData>, delta: Double = AccuracyRunner.delta) {
        for (dataSet in datasets) {
            val (expectedOutputTensors, actualOutputTensors) = dataSet

            val mappedActualOutputTensors = actualOutputTensors.associateBy { it.info.name }

            for (expectedOutputTensor in expectedOutputTensors) {
                val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.info.name] ?: error("Required tensor not found")
                Assertions.assertEquals(expectedOutputTensor, actualOutputTensor, delta)
            }
        }
    }
}
