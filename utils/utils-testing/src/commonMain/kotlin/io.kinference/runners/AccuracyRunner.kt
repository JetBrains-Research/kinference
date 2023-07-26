package io.kinference.runners

import io.kinference.*
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.utils.*
import io.kinference.utils.Assertions.assertLessOrEquals
import okio.Path
import okio.Path.Companion.toPath
import kotlin.math.pow
import kotlin.test.*

class AccuracyRunner<T : ONNXData<*, *>>(private val testEngine: TestEngine<T>) {
    private data class ONNXTestData<T : ONNXData<*, *>> (val name: String, val actual: Map<String, T>, val expected: Map<String, T>)

    data class ONNXTestDataInfo(val path: String, val type: ONNXDataType) {
        companion object {
            private const val DEFAULT_DATATYPE = "ONNX_TENSOR"
            private const val DESC_TYPE_DELIMITER = ":ONNX_TYPE:"

            fun fromString(str: String): ONNXTestDataInfo {
                val split = str.split(DESC_TYPE_DELIMITER)
                val strType = split.getOrElse(1) { DEFAULT_DATATYPE }
                return ONNXTestDataInfo(path = split[0], type = ONNXDataType.fromString(strType))
            }
        }
    }

    private fun List<T>.getInMemorySize() = sumOf { testEngine.getInMemorySize(it) }

    suspend fun runFromS3(name: String, delta: Double = DELTA, disableTests: List<String> = emptyList()) {
        val toFolder = name.replace(":", "/").toPath()
        runTestsFromFolder(S3TestDataLoader, toFolder, disableTests, delta)
    }

    suspend fun runFromResources(testPath: String, delta: Double = DELTA, disableTests: List<String> = emptyList()) {
        runTestsFromFolder(ResourcesTestDataLoader, testPath.toPath(), disableTests, delta)
    }

    private suspend fun runTestsFromFolder(loader: TestDataLoader, testPath: Path, disableTests: List<String> = emptyList(), delta: Double = DELTA) {

        val model = testEngine.loadModel(loader.getFullPath(testPath) / "model.onnx")

        logger.info { "Predict: $testPath" }
        val filesInfo = loader.text(testPath / "descriptor.txt").lines().map { ONNXTestDataInfo.fromString(it) }
        val testGroups = filesInfo.filter { "test" in it.path }.groupBy { info -> info.path.takeWhile { it != '/' } }
        for ((group, files) in testGroups) {
            if (group in disableTests) {
                continue
            }

            val inputFiles = files.filter { file -> "input" in file.path }
            val inputs = inputFiles.map { testEngine.loadData(loader.bytes(testPath / it.path), it.type) }

            val outputFiles =  files.filter { file -> "output" in file.path }
            val expectedOutputs = outputFiles.map { testEngine.loadData(loader.bytes(testPath / it.path), it.type) }

            logger.info { "Start predicting: $group" }
            val actualOutputs: Map<String, T> = if (testEngine is MemoryProfileable) {
                val memoryBeforeTest = testEngine.allocatedMemory()
                logger.info { "Memory before predict: $memoryBeforeTest" }
                val outputs = model.predict(inputs)
                val memoryAfterTest = testEngine.allocatedMemory()
                logger.info { "Memory after predict: $memoryAfterTest" }
                assertLessOrEquals(expectedOutputs.getInMemorySize(), memoryAfterTest - memoryBeforeTest, "Memory leak found")
                outputs
            } else {
                model.predict(inputs)
            }

            check(ONNXTestData(group, expectedOutputs.associateBy { it.name!! }, actualOutputs), delta)

            inputs.forEach { it.close() }
            expectedOutputs.forEach { it.close() }
            actualOutputs.values.forEach { it.close() }
        }
        model.close()
        if (testEngine is MemoryProfileable) {
            assertEquals(0, testEngine.allocatedMemory(), "Memory leak found after model dispose")
        }
    }

    private fun check(dataset: ONNXTestData<T>, delta: Double = DELTA) {
        logger.info { "Dataset: ${dataset.name}\n" }

        val (_, expectedOutputs, actualOutputs) = dataset

        for ((outputName, outputData) in expectedOutputs) {
            val actualOutput = actualOutputs[outputName] ?: error("Required tensor not found")
            testEngine.checkEquals(outputData, actualOutput, delta)
        }
    }

    companion object {
        private val logger = TestLoggerFactory.create("AccuracyRunner")

        private val DELTA = (10.0).pow(-3)
        const val QUANT_DELTA = 3.0
    }
}
