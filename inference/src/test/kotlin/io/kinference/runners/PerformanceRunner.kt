package io.kinference.runners

import io.kinference.data.tensors.Tensor
import io.kinference.model.Model
import io.kinference.utils.DataLoader
import io.kinference.utils.S3Client
import java.io.File

object PerformanceRunner {
    private val testData = File("build/test-data")

    data class PerformanceResults(val name: String, val avg: Double, val min: Long, val max: Long)

    private fun runPerformanceFromS3(testPath: String, prefix: String, count: Int = 10): List<PerformanceResults> {
        val toFolder = File(testData, testPath)
        S3Client.copyObjects(prefix, toFolder)
        return runPerformanceFromFolder(toFolder, count)
    }

    private fun runPerformanceFromResources(testPath: String, count: Int = 10): List<PerformanceResults> {
        val path = javaClass.getResource(testPath)!!.path
        return runPerformanceFromFolder(File(path), count)
    }

    data class TensorDataWithName(val tensors: List<Tensor>, val test: String)

    private fun runPerformanceFromFolder(path: File, count: Int = 10): List<PerformanceResults> {
        val model = Model.load(File(path, "model.onnx").absolutePath)
        val datasets = path.list()!!.filter { "test" in it }.map {
            val inputFiles = File("$path/$it").walk().filter { file -> "input" in file.name }
            val inputTensors = inputFiles.map { DataLoader.getTensor(it) }.toList()
            TensorDataWithName(inputTensors, it)
        }

        val results = ArrayList<PerformanceResults>()

        for (dataset in datasets) {
            val times = LongArray(count)
            for (i in (0 until count)) {
                val startTime = System.currentTimeMillis()
                model.predict(dataset.tensors)
                val endTime = System.currentTimeMillis()
                times[i] = endTime - startTime
            }
            results.add(PerformanceResults(dataset.test, times.average(), times.min()!!, times.max()!!))
        }

        return results
    }

    fun runFromS3(testPath: String, prefix: String, count: Int = 10) {
        output(runPerformanceFromS3(testPath, prefix, count))
    }

    fun runFromResources(testPath: String, count: Int = 10) {
        output(runPerformanceFromResources(testPath, count))
    }

    private fun output(results: List<PerformanceResults>) {
        for (result in results) {
            println("Test ${result.name}: avg ${result.avg}, min ${result.min}, max ${result.max}")
        }
    }
}
