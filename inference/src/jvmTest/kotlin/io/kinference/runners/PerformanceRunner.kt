package io.kinference.runners

import io.kinference.data.tensors.Tensor
import io.kinference.model.Model
import io.kinference.utils.*

object PerformanceRunner {
    data class PerformanceResults(val name: String, val avg: Double, val min: Long, val max: Long)

    private suspend fun runPerformanceFromS3(name: String, count: Int = 10): List<PerformanceResults> {
        val toFolder = name.replace(":", "/")
        return runPerformanceFromFolder(S3TestDataLoader, toFolder, count)
    }

    private suspend fun runPerformanceFromResources(testPath: String, count: Int = 10): List<PerformanceResults> {
        val path = javaClass.getResource(testPath)!!.path
        return runPerformanceFromFolder(ResourcesTestDataLoader, path, count)
    }

    data class TensorDataWithName(val tensors: List<Tensor>, val test: String)

    private suspend fun runPerformanceFromFolder(loader: TestDataLoader, path: String, count: Int = 10): List<PerformanceResults> {
        val model = Model.load(loader.bytes(TestDataLoader.Path(path, "model.onnx")))
        val files = loader.text(TestDataLoader.Path(path, "descriptor.txt")).lines()
        val datasets = files.filter { "test" in it }.groupBy { file -> file.takeWhile { it != '/' } }.map { (group, files) ->
            val inputFiles = files.filter { file -> "input" in file }
            val inputTensors = inputFiles.map { DataLoader.getTensor(loader.bytes(TestDataLoader.Path(path, it))) }.toList()
            TensorDataWithName(inputTensors, group)
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
            results.add(PerformanceResults(dataset.test, times.average(), times.minOrNull()!!, times.max()!!))
        }

        return results
    }

    suspend fun runFromS3(name: String, count: Int = 20) {
        output(runPerformanceFromS3(name, count))
    }

    suspend fun runFromResources(testPath: String, count: Int = 20) {
        output(runPerformanceFromResources(testPath, count))
    }

    private fun output(results: List<PerformanceResults>) {
        for (result in results.sortedBy { it.name }) {
            println("Test ${result.name}: avg ${result.avg}, min ${result.min}, max ${result.max}")
        }
    }
}
