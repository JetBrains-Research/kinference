package io.kinference.core

import io.kinference.model.*
import io.kinference.core.model.KIModel
import kotlinx.benchmark.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.*
import kotlinx.coroutines.runBlocking
import io.kinference.core.*
import okio.Path.Companion.toPath
import io.kinference.utils.*
import io.kinference.runners.*
import io.kinference.utils.PredictionConfigs.DefaultAutoAllocator
import io.kinference.utils.PredictionConfigs.DefaultManualAllocator
import io.kinference.utils.PredictionConfigs.NoAllocator


@State(Scope.Benchmark)
open class Electra {
    val engine = KIEngine
    val loader = S3TestDataLoader
    lateinit var model: KIModel
    lateinit var inputs: List<List<KIONNXData<*>>>

    val predictionConfigs = mapOf(
        "manual" to DefaultManualAllocator, "auto" to DefaultAutoAllocator, "none" to NoAllocator
    )

    @Param("manual", "auto", "none")
    lateinit var configName: String

    @Param("1", "2", "5")
    var runs = 1

    @Setup
    fun loadModel() {
        runBlocking {
            val path = "bert/electra".toPath()
            val cfg = predictionConfigs[configName]!!
            model = engine.loadModel(loader.getFullPath(path / "model.onnx"), optimize = true, predictionConfig = cfg)
            val fileInfo = loader.text(path / "descriptor.txt").lines().map { AccuracyRunner.ONNXTestDataInfo.fromString(it) }
            val datasets = fileInfo.filter { "test" in it.path }.groupBy { info -> info.path.takeWhile { it != '/' } }.map { (group, files) ->
                val inputFiles = files.filter { file -> "input" in file.path }
                val inputs = inputFiles.map { loader.bytes(path / it.path) to it.type }
                PerformanceRunner.ONNXDataWithName(inputs, group)
            }
            inputs = datasets.map { x -> x.data.map { engine.loadData(it.first, it.second) } }
        }
    }

    @Benchmark
    fun run(bh: Blackhole) {
        runBlocking {
            repeat(runs) {
                for (input in inputs)
                    bh.consume(model.predict(input))
            }
        }
    }
}
