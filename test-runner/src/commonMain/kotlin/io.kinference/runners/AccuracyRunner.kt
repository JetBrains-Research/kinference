package io.kinference.runners

import io.kinference.TestEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.logger
import io.kinference.utils.*
import kotlin.math.pow
import kotlin.time.ExperimentalTime

@ExperimentalTime
class AccuracyRunner(private val testEngine: TestEngine) {
    data class ONNXTestData(val name: String, val actual: List<ONNXData<*>>, val expected: List<ONNXData<*>>)
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

    private suspend fun runTestsFromS3(name: String, disableTests: List<String> = emptyList()): List<ONNXTestData> {
        val toFolder = name.replace(":", "/")
        return runTestsFromFolder(S3TestDataLoader, toFolder, disableTests)
    }

    private suspend fun runTestsFromResources(testPath: String, disableTests: List<String> = emptyList()): List<ONNXTestData> {
        val path = "build/processedResources/${TestRunner.forPlatform("js", "jvm")}/test/${testPath}"
        return runTestsFromFolder(ResourcesTestDataLoader, path, disableTests)
    }

    private suspend fun runTestsFromFolder(loader: TestDataLoader, path: String, disableTests: List<String> = emptyList()): List<ONNXTestData> {
        val model = testEngine.loadModel(loader.bytes(TestDataLoader.Path(path, "model.onnx")))
        val filesInfo = loader.text(TestDataLoader.Path(path, "descriptor.txt")).lines().map { ONNXTestDataInfo.fromString(it) }
        return filesInfo.filter { "test" in it.path }.groupBy { info -> info.path.takeWhile { it != '/' } }.map { (group, files) ->
            if (group in disableTests) {
                null
            } else {
                val inputFiles = files.filter { file -> "input" in file.path }
                val inputs = inputFiles.map { testEngine.loadData(loader.bytes(TestDataLoader.Path(path, it.path)), it.type) }

                val outputFiles =  files.filter { file -> "output" in file.path }
                val expectedOutputs = outputFiles.map { testEngine.loadData(loader.bytes(TestDataLoader.Path(path, it.path)), it.type) }.toList()

                logger.info { "Start predicting: $group" }
                val actualOutputs = model.predict(inputs)
                ONNXTestData(group, expectedOutputs, actualOutputs)
            }
        }.filterNotNull()
    }

    suspend fun runFromS3(name: String, delta: Double = DELTA, disableTests: List<String> = emptyList()) {
        check(runTestsFromS3(name, disableTests), delta)
    }

    suspend fun runFromResources(path: String, delta: Double = DELTA, disableTests: List<String> = emptyList()) {
        check(runTestsFromResources(path, disableTests), delta)
    }

    private fun check(datasets: List<ONNXTestData>, delta: Double = DELTA) {
        for (dataSet in datasets) {
            logger.info { "Dataset: ${dataSet.name}\n" }

            val (_, expectedOutputs, actualOutputs) = dataSet

            val mappedActualOutputs = actualOutputs.associateBy { it.name }

            for (expectedOutput in expectedOutputs) {
                val actualOutput = mappedActualOutputs[expectedOutput.name] ?: error("Required tensor not found")
                testEngine.checkEquals(expectedOutput, actualOutput, delta)
            }
        }
    }

    companion object {
        val logger = logger("Runner")

        private val DELTA = (10.0).pow(-3)
        const val QUANT_DELTA = 3.0
    }
}
