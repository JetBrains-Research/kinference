package io.kinference.model

import io.kinference.OptimizableEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.optimizer.OptimizerRule
import okio.Path

/**
 * This interface defines an inference engine with model optimization rules accessible.
 * The engine is unique for every supported KInference backend.
 */
interface IrOptimizableEngine<T : ONNXData<*, *>> : OptimizableEngine<T> {
    /**
     * Reads model of given [ONNXDataType] and applies optimization rules from [rules] list to it.
     * Model should be previously loaded as [ByteArray].
     */
    suspend fun loadModel(bytes: ByteArray, rules: List<OptimizerRule<T>> = emptyList()): Model<T>

    /**
     * Reads model from the specified path and applies optimization rules from [rules] list to it.
     */
    suspend fun loadModel(path: Path, rules: List<OptimizerRule<T>> = emptyList()): Model<T>

    /**
     * Reads model from the specified string path and applies optimization rules from [rules] list to it.
     */
    suspend fun loadModel(path: String, rules: List<OptimizerRule<T>> = emptyList()): Model<T>
}
