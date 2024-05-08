package io.kinference.runners

import io.kinference.TestEngine
import io.kinference.TestLoggerFactory
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import io.kinference.profiler.Profilable
import io.kinference.utils.*
import io.kinference.utils.time.Timer
import kotlinx.coroutines.*
import okio.Path
import okio.Path.Companion.toPath

class PerformanceRunner<T : ONNXData<*, *>>(private val engine: TestEngine<T>) {
    data class PerformanceResults(val name: String, val avg: Double, val min: Long, val max: Long)

    private suspend fun runPerformanceFromS3(name: String, count: Int = 10, warmup: Int = 3, withProfiling: Boolean = false, parallelLoad: Boolean = false): List<PerformanceResults> {
        val toFolder = name.replace(":", "/").toPath()
        return runPerformanceFromFolder(S3TestDataLoader, toFolder, count, warmup, withProfiling, parallelLoad)
    }

    private suspend fun runPerformanceFromResources(testPath: String, count: Int = 10, warmup: Int = 3, withProfiling: Boolean = false, parallelLoad: Boolean = false): List<PerformanceResults> {
        return runPerformanceFromFolder(ResourcesTestDataLoader, testPath.toPath(), count, warmup, withProfiling, parallelLoad)
    }

//    data class ONNXDataWithName(val data: Map<String, ONNXData<*>>, val test: String)
    data class ONNXDataWithName(val data: Collection<Pair<ByteArray, ONNXDataType>>, val test: String)

    private suspend fun runPerformanceFromFolder(
        loader: TestDataLoader,
        path: Path,
        count: Int = 10,
        warmup: Int = 3,
        withProfiling: Boolean = false,
        parallelLoad: Boolean = false
    ): List<PerformanceResults> {
        logger.info { "Predict: $path" }

        lateinit var model: Model<T>
        val modelLoadTime = Timer.measure {
            model = engine.loadModel(loader.getFullPath(path / "model.onnx"))
        }
        logger.info { "Model load time: $modelLoadTime" }

        val fileInfo = loader.text(path / "descriptor.txt").lines().map { AccuracyRunner.ONNXTestDataInfo.fromString(it) }
        val datasets = fileInfo.filter { "test" in it.path }.groupBy { info -> info.path.takeWhile { it != '/' } }.map { (group, files) ->
            val inputFiles = files.filter { file -> "input" in file.path }
            val inputs = inputFiles.map { loader.bytes(path / it.path) to it.type }
            ONNXDataWithName(inputs, group)
        }

        val results = ArrayList<PerformanceResults>()

        for (dataset in datasets) {
            val inputs = dataset.data.map { engine.loadData(it.first, it.second) }

            repeat(warmup) {
                val outputs = model.predict(inputs)
                outputs.values.forEach { it.close() }
            }

            val times = LongArray(count)

            if (parallelLoad) {
                val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
                val predictions: List<Deferred<Map<String, T>>> = List(count) { i ->
                    scope.async {
                        lateinit var outputs: Map<String, T>
                        val time = Timer.measure {
                            outputs = model.predict(inputs, withProfiling)
                        }.millis
                        times[i] = time
                        outputs
                    }
                }

                val res = predictions.awaitAll()
                res.forEach { output ->
                    output.values.forEach { it.close() }
                }
            } else {
                for (i in (0 until count)) {
                    lateinit var outputs: Map<String, T>
                    val time = Timer.measure {
                        outputs = model.predict(inputs, withProfiling)
                    }.millis
                    times[i] = time

                    outputs.values.forEach { it.close() }
                }
            }

            results.add(PerformanceResults(dataset.test, times.average(), times.minOrNull()!!, times.maxOrNull()!!))

            if (withProfiling && model is Profilable) {
                logger.info {
                    "Results for ${dataset.test}:" +
                        (model as Profilable).analyzeProfilingResults().getInfo()
                }

                (model as Profilable).resetProfiles()
            }

            inputs.forEach { it.close() }
        }

        model.close()
        return results
    }

    suspend fun runFromS3(name: String, count: Int = 20, warmup: Int = 3, withProfiling: Boolean = false, parallelLoad: Boolean = false) {
        output(runPerformanceFromS3(name, count, warmup, withProfiling, parallelLoad))
    }

    suspend fun runFromResources(testPath: String, count: Int = 20, warmup: Int = 3, withProfiling: Boolean = false, parallelLoad: Boolean = false) {
        output(runPerformanceFromResources(testPath, count, warmup, withProfiling, parallelLoad))
    }

    private fun output(results: List<PerformanceResults>) {
        for (result in results.sortedBy { it.name }) {
            logger.info { "Test ${result.name}: avg ${result.avg}, min ${result.min}, max ${result.max}" }
        }
    }

    companion object {
        private val logger = TestLoggerFactory.create("io.kinference.runners.PerformanceRunner")
    }
}
