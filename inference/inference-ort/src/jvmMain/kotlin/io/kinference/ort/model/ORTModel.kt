package io.kinference.ort.model

import ai.onnxruntime.*
import io.kinference.model.Model
import io.kinference.ort.ORTData
import io.kinference.ort.data.utils.createORTData
import io.kinference.utils.LoggerFactory

class ORTModel(private val session: OrtSession) : Model<ORTData<*>> {
    override fun predict(input: Map<String, ORTData<*>>, profile: Boolean): Map<String, ORTData<*>> {
        if (profile) logger.warning { "Profiling of models running on OnnxRuntime backend is not supported" }
        val inputsMap = input.mapValues { it.value.data as OnnxTensor }
        val outputMap = session.run(inputsMap)
        return outputMap.associate { it.key to createORTData(it.key, it.value) }
    }

    companion object {
        private val logger = LoggerFactory.create("io.kinference.ort.model.ORTModel")
    }
}
