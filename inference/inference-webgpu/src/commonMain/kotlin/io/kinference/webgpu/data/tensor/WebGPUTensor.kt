package io.kinference.webgpu.data.tensor

import io.kinference.data.ONNXData
import io.kinference.data.ONNXTensor
import io.kinference.ndarray.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.types.ValueTypeInfo
import io.kinference.webgpu.engine.WebGPUBackend
import io.kinference.webgpu.ndarray.*

class WebGPUTensor(name: String?, data: NDArray, val info: ValueTypeInfo.TensorTypeInfo) : ONNXTensor<NDArray, WebGPUBackend>(name, data) {
    override val backend = WebGPUBackend

    override fun rename(name: String): ONNXData<NDArray, WebGPUBackend> = WebGPUTensor(name, data, info)

    companion object {
        @Suppress("UNCHECKED_CAST")
        fun create(proto: TensorProto): WebGPUTensor {
            val type = proto.dataType ?: TensorProto.DataType.UNDEFINED
            val array = parseArray(proto)
            requireNotNull(array) { "Array value should be initialized" }

            return WebGPUTensor(array, type, proto.dims, proto.name)
        }

        private operator fun invoke(value: Any, type: TensorProto.DataType, shape: IntArray, name: String? = ""): WebGPUTensor {
            val nameNotNull = name.orEmpty()
            return when (type) {
                TensorProto.DataType.BOOL -> NDArray.intNDArray(NDArrayInfo(shape, WebGPUDataType.INT32), (value as BooleanArray).toIntArray()).asTensor(nameNotNull)
                TensorProto.DataType.INT8 -> NDArray.intNDArray(NDArrayInfo(shape, WebGPUDataType.INT32), (value as ByteArray).toIntArray()).asTensor(nameNotNull)
                TensorProto.DataType.INT16 -> NDArray.intNDArray(NDArrayInfo(shape, WebGPUDataType.INT32), (value as ShortArray).toIntArray()).asTensor(nameNotNull)
                TensorProto.DataType.INT32 -> NDArray.intNDArray(NDArrayInfo(shape, WebGPUDataType.INT32), value as IntArray).asTensor(nameNotNull)
                TensorProto.DataType.INT64 -> NDArray.intNDArray(NDArrayInfo(shape, WebGPUDataType.INT32), (value as LongArray).toIntArray()).asTensor(nameNotNull)
                TensorProto.DataType.UINT8 -> NDArray.uintNDArray(NDArrayInfo(shape, WebGPUDataType.UINT32), (value as UByteArray).toUIntArray()).asTensor(nameNotNull)
                TensorProto.DataType.UINT16 -> NDArray.uintNDArray(NDArrayInfo(shape, WebGPUDataType.UINT32), (value as UShortArray).toUIntArray()).asTensor(nameNotNull)
                TensorProto.DataType.UINT32 -> NDArray.uintNDArray(NDArrayInfo(shape, WebGPUDataType.UINT32), value as UIntArray).asTensor(nameNotNull)
                TensorProto.DataType.UINT64 -> NDArray.uintNDArray(NDArrayInfo(shape, WebGPUDataType.UINT32), (value as ULongArray).toUIntArray()).asTensor(nameNotNull)
                TensorProto.DataType.FLOAT -> NDArray.floatNDArray(NDArrayInfo(shape, WebGPUDataType.FLOAT32), value as FloatArray).asTensor(nameNotNull)
                TensorProto.DataType.DOUBLE -> NDArray.floatNDArray(NDArrayInfo(shape, WebGPUDataType.FLOAT32), (value as DoubleArray).toFloatArray()).asTensor(nameNotNull)
                else -> error("Unsupported type")
            }
        }

        private fun parseArray(proto: TensorProto) = when {
            proto.isString() -> proto.stringData
            proto.isPrimitive() -> proto.arrayData
            else -> error("Unsupported data type ${proto.dataType}")
        }
    }
}
