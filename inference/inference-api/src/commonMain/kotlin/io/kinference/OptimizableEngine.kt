package io.kinference

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import okio.Path

/**
 * This interface defines inference engine with model optimization options.
 * The engine is unique for every supported KInference backend.
 */
interface OptimizableEngine<T : ONNXData<*, *>> : InferenceEngine<T> {
    /**
     * Reads model of given [ONNXDataType].
     * Model should be previously loaded as [ByteArray].
     * If [optimize] flag is true, runs available optimizations on the given model.
     */
    suspend fun loadModel(bytes: ByteArray, optimize: Boolean = false): Model<T>

    /**
     * Reads model from the specified path.
     * If [optimize] flag is true, runs available optimizations on the given model.
     */
    suspend fun loadModel(path: Path, optimize: Boolean = false): Model<T>

    /**
     * Reads model from the specified string path.
     * If [optimize] flag is true, runs available optimizations on the given model.
     */
    suspend fun loadModel(path: String, optimize: Boolean = false): Model<T>
}
