package io.kinference.ort.data

import ai.onnxruntime.*
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.extensions.primitiveFromTiledArray
import io.kinference.protobuf.message.TensorProto
import java.nio.*

abstract class ORTData(override val data: OnnxValue, override val name: String?) : ONNXData<OnnxValue> {
    companion object {
        inline operator fun <reified V : OnnxValue> invoke(name: String?, data: V) : ORTData = when (data) {
            is OnnxTensor -> ORTTensor(data, name)
            is OnnxMap -> ORTMap(data, name)
            is OnnxSequence -> ORTSequence(data, name)
            else -> error("")
        }
    }
}

class ORTSequence(override val data: OnnxSequence, name: String?) : ORTData(data, name) {
    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
    override fun rename(name: String): ORTData = ORTSequence(data, name)
}

class ORTMap(override val data: OnnxMap, name: String?) : ORTData(data, name) {
    override val type: ONNXDataType = ONNXDataType.ONNX_MAP
    override fun rename(name: String): ORTData = ORTMap(data, name)
}

class ORTTensor(override val data: OnnxTensor, name: String?) : ORTData(data, name) {
    override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
    override fun rename(name: String): ORTData = ORTTensor(data, name)

    companion object {
        fun create(proto: TensorProto): ORTTensor {
            val type = proto.dataType ?: TensorProto.DataType.UNDEFINED
            val array = when {
                proto.isTiled() -> primitiveFromTiledArray(proto.arrayData!!)
                proto.isString() -> proto.stringData
                proto.isPrimitive() -> proto.arrayData
                else -> error("Unsupported data type ${proto.dataType}")
            }
            requireNotNull(array) { "Array value should be initialized" }

            return ORTTensor(array, type, proto.dims.toLongArray(), proto.name)
        }

        private fun IntArray.toLongArray() = LongArray(this.size) { this[it].toLong() }

        private operator fun invoke(value: Any, type: TensorProto.DataType, dims: LongArray = LongArray(0), name: String? = null): ORTTensor {
            val onnxTensor = when (type) {
                TensorProto.DataType.DOUBLE -> {
                    val buffer = DoubleBuffer.wrap(value as DoubleArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.FLOAT -> {
                    val buffer = FloatBuffer.wrap(value as FloatArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.INT32 -> {
                    val buffer = IntBuffer.wrap(value as IntArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.INT8 -> {
                    val buffer = ByteBuffer.wrap(value as ByteArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.INT64 -> {
                    val buffer = LongBuffer.wrap(value as LongArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.INT16 -> {
                    val buffer = ShortBuffer.wrap(value as ShortArray)
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, dims)
                }
                TensorProto.DataType.STRING -> {
                    val array = (value as List<String>).toTypedArray()
                    OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), array, dims)
                }
                else -> error("Unsupported data type $type")
            }
            return ORTTensor(onnxTensor, name ?: "")
        }
    }
}
