package io.kinference

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import okio.Path

abstract class BackendInfo(val name: String)

/**
 * This interface defines an inference engine.
 * Inference engine manages how models and ONNX data should be loaded and processed for further usage.
 * The engine is unique for every supported KInference backend.
 */
interface InferenceEngine<T : ONNXData<*, *>> {
    /**
     * Represents the backend this engine runs on.
     */
    val info: BackendInfo

    /**
     * Reads data of a given type. Data should be previously loaded as [ByteArray].
     */
    suspend fun loadData(bytes: ByteArray, type: ONNXDataType): T

    /**
     * Reads model. Model should be previously loaded as [ByteArray].
     */
    suspend fun loadModel(bytes: ByteArray): Model<T>

    /**
     * Reads model from the specified path.
     */
    suspend fun loadModel(path: Path): Model<T>

    /**
     * Reads data of a given type from the specified path.
     */
    suspend fun loadData(path: Path, type: ONNXDataType): T

    /**
     * Reads model from the specified string path.
     */
    suspend fun loadModel(path: String): Model<T>

    /**
     * Reads data of a given type from the specified string path.
     */
    suspend fun loadData(path: String, type: ONNXDataType): T
}
