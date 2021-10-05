package io.kinference.runners

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.data.tensors.Tensor
import io.kinference.model.Model
import io.kinference.ndarray.logger
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.*
import kotlin.math.pow
import kotlin.test.assertEquals
import kotlin.time.ExperimentalTime

@ExperimentalTime
object AccuracyRunner {
    val logger = logger("Runner")

    private val delta = (10.0).pow(-3)

    const val QUANT_DELTA = 3.0

    data class ONNXTestData(val name: String, val actual: List<ONNXData>, val expected: List<ONNXData>)
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

    suspend fun runFromS3(name: String, delta: Double = AccuracyRunner.delta, disableTests: List<String> = emptyList()) {
        val toFolder = name.replace(":", "/")
        runTestsFromFolder(S3TestDataLoader, toFolder, delta, disableTests)
    }

    suspend fun runFromResources(testPath: String, delta: Double = AccuracyRunner.delta, disableTests: List<String> = emptyList()) {
        val path = "build/processedResources/js/test/${testPath}"
        runTestsFromFolder(ResourcesTestDataLoader, path, delta, disableTests)
    }

    private suspend fun runTestsFromFolder(loader: TestDataLoader, path: String,
                                           delta: Double = AccuracyRunner.delta,
                                           disableTests: List<String> = emptyList()) {
        val model = Model.load(loader.bytes(TestDataLoader.Path(path, "model.onnx")))
        val filesInfo = loader.text(TestDataLoader.Path(path, "descriptor.txt")).lines().map { ONNXTestDataInfo.fromString(it) }
        val datasetsFiles = filesInfo.filter { "test" in it.path }.groupBy { info -> info.path.takeWhile { it != '/' } }
        for ((group, files) in datasetsFiles) {
            if (group in disableTests) {
                continue
            } else {
                val inputFiles = files.filter { file -> "input" in file.path }
                val inputProtos = inputFiles.map { TensorProto.decode(loader.bytes(TestDataLoader.Path(path, it.path))) }
                val inputs = inputProtos.map{ model.graph.prepareInput(it) }

                val outputFiles =  files.filter { file -> "output" in file.path }
                val expectedOutputs = outputFiles.map { DataLoader.getData(loader.bytes(TestDataLoader.Path(path, it.path)), it.type) }.toList()

                logger.info { "Start predicting: $group" }
                val actualOutputs = model.predict(inputs)
                check(ONNXTestData(group, expectedOutputs, actualOutputs), delta)
                inputs.forEach { it.data.dispose() }
                expectedOutputs.forEach {
                    if (it is Tensor) {
                        it.data.dispose()
                    }
                }
                actualOutputs.forEach {
                    if (it is Tensor) {
                        it.data.dispose()
                    }
                }
            }
        }
    }

    private fun check(dataSet: ONNXTestData, delta: Double = AccuracyRunner.delta) {
        logger.info { "Dataset: ${dataSet.name}\n" }

        val (_, expectedOutputTensors, actualOutputTensors) = dataSet

        val mappedActualOutputTensors = actualOutputTensors.associateBy { it.info.name }

        for (expectedOutputTensor in expectedOutputTensors) {
            val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.info.name] ?: error("Required tensor not found")
            Assertions.assertEquals(expectedOutputTensor, actualOutputTensor, delta)
        }
    }
}
