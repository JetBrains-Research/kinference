package io.kinference.runners

import io.kinference.data.ONNXData
import io.kinference.model.Model
import io.kinference.model.load
import io.kinference.onnx.TensorProto
import io.kinference.utils.*
import java.io.File
import kotlin.math.pow

object AccuracyRunner {
    private val testData = File("../build/test-data")

    private val delta = (10.0).pow(-3)

    data class ONNXTestData(val actual: List<ONNXData>, val expected: List<ONNXData>)

    private suspend fun runTestsFromS3(name: String, testRunner: suspend (File) -> List<ONNXTestData>): List<ONNXTestData> {
        val toFolder = File(testData, "tests/${name.replace(":", "/")}/")
        return testRunner(toFolder)
    }

    private suspend fun runTestsFromResources(testPath: String): List<ONNXTestData> {
        val path = javaClass.getResource(testPath)!!.path
        return runTestsFromFolder(File(path))
    }

    private suspend fun runTestsFromFolder(path: File): List<ONNXTestData> {
        val model = Model.load(TestResourceLoader.fileBytes("$path/model.onnx"))
        val files = TestResourceLoader.fileText("$path/descriptor.txt").lines().map { it.drop(path.absolutePath.length) }
        return files.filter { "test" in it }.groupBy { file -> file.takeWhile { it != '/' } }.map { (group, files) ->
            val inputFiles = files.filter { file -> "input" in file }
            val inputTensors = inputFiles.map { DataLoader.getTensor(TestResourceLoader.fileBytes(it)) }.toList()

            val outputFiles =  files.filter { file -> "output" in file }
            val expectedOutputTensors = outputFiles.map { DataLoader.getTensor(TestResourceLoader.fileBytes(it)) }.toList()

            val actualOutputTensors = model.predict(inputTensors)
            ONNXTestData(expectedOutputTensors, actualOutputTensors)
        }
    }

    suspend fun runFromS3(name: String, delta: Double = AccuracyRunner.delta) {
        check(runTestsFromS3(name, this::runTestsFromFolder), delta)
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
