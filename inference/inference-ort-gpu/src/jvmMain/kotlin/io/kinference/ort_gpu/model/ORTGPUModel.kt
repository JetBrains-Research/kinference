package io.kinference.ort_gpu.model

import ai.onnxruntime.*
import io.kinference.model.ExecutionContext
import io.kinference.model.Model
import io.kinference.ort_gpu.ORTGPUData
import io.kinference.ort_gpu.data.utils.createORTData
import io.kinference.utils.LoggerFactory

class ORTGPUModel(private val session: OrtSession) : Model<ORTGPUData<*>> {
    override fun predict(input: List<ORTGPUData<*>>, profile: Boolean, executionContext: ExecutionContext?): Map<String, ORTGPUData<*>> {
        if (profile) logger.warning { "Profiling of models running on OnnxRuntime backend is not supported" }
        if (executionContext != null) logger.warning { "ExecutionContext for models running on OnnxRuntime backend is not supported" }
        val inputsMap = input.associate { it.name to it.data as OnnxTensor }
        val outputMap = session.run(inputsMap)
        return outputMap.associate { it.key to createORTData(it.key, it.value) }
    }

    companion object {
        private val logger = LoggerFactory.create("io.kinference.ort.model.ORTModel")
    }
}
