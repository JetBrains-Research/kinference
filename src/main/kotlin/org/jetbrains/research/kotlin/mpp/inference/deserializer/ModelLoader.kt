package org.jetbrains.research.kotlin.mpp.inference.deserializer

import io.jhdf.HdfFile
import io.jhdf.api.Dataset
import org.jetbrains.research.kotlin.mpp.inference.deserializer.json.ModelConfig
import org.jetbrains.research.kotlin.mpp.inference.deserializer.json.ModelScheme
import org.jetbrains.research.kotlin.mpp.inference.nn.layer.Layer
import org.jetbrains.research.kotlin.mpp.inference.nn.model.sequential.Perceptron
import java.io.File

@Suppress("UNCHECKED_CAST")
object ModelLoader {
    fun loadPerceptronModel(model: File): Perceptron {
        val (config, layers) = importSequentialModelParameters(model)
        config as ModelConfig.Sequential
        return Perceptron.create(config.name, layers, config.batchInputShape)
    }

    private fun importSequentialModelParameters(model: File): Pair<ModelConfig, List<Layer<*>>> {
        val hdf = HdfFile(model)

        val config = importModelConfig(hdf)
        val layers = importSequentialModelLayers(hdf, config.config)

        return config.config to layers
    }

    private fun importSequentialModelLayers(hdf: HdfFile, config: ModelConfig) = config.layers.mapNotNull { layer ->
        val layerWeights = hdf.getByPath("model_weights/${layer.config.name}")
        val weightNames = (layerWeights.getAttribute("weight_names").data as? Array<String?>)?.filterNotNull()

        val (weights, biases) = weightNames?.map { weight ->
            val data = hdf.getByPath("model_weights/${layer.config.name}/$weight") as Dataset
            weight to data.data.toMatrix()
        }?.partition { !it.first.contains("bias") } ?: null to null

        val params = Layer.Parameters(
            weights?.map { it.second }?.singleOrNull(),
            biases?.map { it.second }?.singleOrNull()
        )
        Layer.create(layer, params)
    }

    private fun importModelConfig(hdf: HdfFile): ModelScheme {
        val configString = hdf.getAttribute("model_config").data as String
        return ModelScheme.parse(configString)
    }
}

