package io.kinference.tfjs

import io.kinference.BackendInfo
import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.protobuf.*
import io.kinference.protobuf.message.*
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.model.TFJSModel
import io.kinference.utils.CommonDataLoader
import okio.Buffer
import okio.Path
import okio.Path.Companion.toPath

typealias TFJSData<T> = ONNXData<T, TFJSBackend>

object TFJSBackend : BackendInfo(name = "TensorFlow for JS")

/**
 * This is an inference engine for KInference TensorFlow.js backend implementation.
 * High-performance JavaScript backend that is built upon [Tensorflow.js](https://www.tensorflow.org/js/) library.
 * Essentially, it employs GPU operations provided by TensorFlow.js to boost the computations.
 * Recommended backend for JavaScript projects.
 *
 * TensorFlow.js version: 4.0.0
 */
object TFJSEngine : InferenceEngine<TFJSData<*>> {
    override val info: BackendInfo
        get() = TFJSBackend

    private val TFJS_READER_CONFIG = ProtobufReader.ReaderConfig(tensorDecoder = FlatTensorDecoder)
    private fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), TFJS_READER_CONFIG)

    override fun loadModel(bytes: ByteArray): TFJSModel {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        return TFJSModel(modelScheme)
    }

    override suspend fun loadModel(path: Path): TFJSModel {
        return loadModel(CommonDataLoader.bytes(path))
    }

    override suspend fun loadModel(path: String): TFJSModel {
        return loadModel(path.toPath())
    }

    override fun loadData(bytes: ByteArray, type: ONNXDataType): TFJSData<*> {
        return when (type) {
            ONNXDataType.ONNX_TENSOR -> TFJSTensor.create(protoReader(bytes).readTensor())
            ONNXDataType.ONNX_SEQUENCE -> TFJSSequence.create(SequenceProto.decode(protoReader(bytes)))
            ONNXDataType.ONNX_MAP -> TFJSMap.create(MapProto.decode(protoReader(bytes)))
        }
    }

    override suspend fun loadData(path: Path, type: ONNXDataType): TFJSData<*> {
        return loadData(CommonDataLoader.bytes(path), type)
    }

    override suspend fun loadData(path: String, type: ONNXDataType): TFJSData<*> {
        return loadData(path.toPath(), type)
    }
}
