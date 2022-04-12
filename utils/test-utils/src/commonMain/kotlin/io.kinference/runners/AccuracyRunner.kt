package io.kinference.runners

import io.kinference.TestEngine
import io.kinference.TestLoggerFactory
import io.kinference.data.*
import io.kinference.utils.*
import kotlin.math.pow
import kotlin.time.ExperimentalTime

@ExperimentalTime
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

    suspend fun runFromS3(name: String, delta: Double = DELTA, disableTests: List<String> = emptyList()) {
        val toFolder = name.replace(":", "/")
        runTestsFromFolder(S3TestDataLoader, toFolder, disableTests, delta)
    }

    suspend fun runFromResources(testPath: String, delta: Double = DELTA, disableTests: List<String> = emptyList()) {
        val path = "build/processedResources/${TestRunner.forPlatform("jsLegacy", "jvm")}/main/${testPath}"
        runTestsFromFolder(ResourcesTestDataLoader, path, disableTests, delta)
    }

    private suspend fun runTestsFromFolder(loader: TestDataLoader, path: String, disableTests: List<String> = emptyList(), delta: Double = DELTA) {
        val model = testEngine.loadModel(loader.bytes(TestDataLoader.Path(path, "model.onnx")))

        logger.info { "Predict: $path" }
        val filesInfo = loader.text(TestDataLoader.Path(path, "descriptor.txt")).lines().map { ONNXTestDataInfo.fromString(it) }
        val testGroups = filesInfo.filter { "test" in it.path }.groupBy { info -> info.path.takeWhile { it != '/' } }
        for ((group, files) in testGroups) {
            if (group in disableTests) {
                continue
            }

            val inputFiles = files.filter { file -> "input" in file.path }
            val inputs = inputFiles.map { testEngine.loadData(loader.bytes(TestDataLoader.Path(path, it.path)), it.type) }

            val outputFiles =  files.filter { file -> "output" in file.path }
            val expectedOutputs = outputFiles.map { testEngine.loadData(loader.bytes(TestDataLoader.Path(path, it.path)), it.type) }

            logger.info { "Start predicting: $group" }
            val actualOutputs = model.predictSuspend(inputs)
            check(ONNXTestData(group, expectedOutputs.associateBy { it.name!! }, actualOutputs), delta)

            inputs.forEach { testEngine.postprocessData(it) }
            expectedOutputs.forEach { testEngine.postprocessData(it) }
            actualOutputs.values.forEach { testEngine.postprocessData(it) }
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
