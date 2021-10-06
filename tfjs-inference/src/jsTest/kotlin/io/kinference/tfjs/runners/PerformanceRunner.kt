package io.kinference.tfjs.runners

import io.kinference.tfjs.data.ONNXData
import io.kinference.tfjs.data.tensors.Tensor
import io.kinference.tfjs.model.Model
import io.kinference.ndarray.logger
import io.kinference.tfjs.utils.*
import kotlin.time.measureTime

object PerformanceRunner {
    val logger = logger("PerformanceRunner")

    data class PerformanceResults(val name: String, val avg: Double, val min: Long, val max: Long)

    private suspend fun runPerformanceFromS3(name: String, count: Int = 10, withProfiling: Boolean = false): List<PerformanceResults> {
        val toFolder = name.replace(":", "/")
        return runPerformanceFromFolder(S3TestDataLoader, toFolder, count, withProfiling)
    }

    private suspend fun runPerformanceFromResources(testPath: String, count: Int = 10, withProfiling: Boolean = false): List<PerformanceResults> {
        val path = "build/processedResources/js/test/${testPath}"
        return runPerformanceFromFolder(ResourcesTestDataLoader, path, count, withProfiling)
    }

    data class ONNXDataWithName(val data: List<ONNXData>, val test: String)

    private suspend fun runPerformanceFromFolder(loader: TestDataLoader, path: String, count: Int = 10, withProfiling: Boolean = false): List<PerformanceResults> {
        val model = Model.load(loader.bytes(TestDataLoader.Path(path, "model.onnx")))
        val fileInfo = loader.text(TestDataLoader.Path(path, "descriptor.txt")).lines().map { AccuracyRunner.ONNXTestDataInfo.fromString(it) }
        val datasets = fileInfo.filter { "test" in it.path }.groupBy { info -> info.path.takeWhile { it != '/' } }.map { (group, files) ->
            val inputFiles = files.filter { file -> "input" in file.path }
            val inputs = inputFiles.map { DataLoader.getData(loader.bytes(TestDataLoader.Path(path, it.path)), it.type) }.toList()
            ONNXDataWithName(inputs, group)
        }

        val results = ArrayList<PerformanceResults>()

        for (dataset in datasets) {
            val times = LongArray(count)
            repeat(3) {
                val outputTensors = model.predict(dataset.data)

                outputTensors.forEach { if (it is Tensor) it.data.dispose() }
            }

            for (i in (0 until count)) {
                lateinit var outputTensors: List<ONNXData>

                val time = measureTime {
                    outputTensors = model.predict(dataset.data)
                }.inMilliseconds.toLong()
                times[i] = time

                outputTensors.forEach { if (it is Tensor) it.data.dispose() }
            }
            results.add(PerformanceResults(dataset.test, times.average(), times.minOrNull()!!, times.maxOrNull()!!))
        }

        return results
    }

    suspend fun runFromS3(name: String, count: Int = 20, withProfiling: Boolean = false) {
        output(runPerformanceFromS3(name, count, withProfiling))
    }

    suspend fun runFromResources(testPath: String, count: Int = 20, withProfiling: Boolean = false) {
        output(runPerformanceFromResources(testPath, count, withProfiling))
    }

    private fun output(results: List<PerformanceResults>) {
        for (result in results) {
            println("Test ${result.name}: avg ${result.avg}, min ${result.min}, max ${result.max}\n")
        }
    }
}
