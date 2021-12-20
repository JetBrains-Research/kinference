package io.kinference.webgpu.tensor

import io.kinference.data.ONNXData
import io.kinference.data.ONNXTensor
import io.kinference.ndarray.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.types.ValueTypeInfo
import io.kinference.utils.webgpu.Device
import io.kinference.webgpu.engine.WebGPUBackend
import io.kinference.webgpu.ndarray.*

class WebGPUTensor(name: String?, data: NDArray, val info: ValueTypeInfo.TensorTypeInfo) : ONNXTensor<NDArray, WebGPUBackend>(name, data) {
    override val backend = WebGPUBackend

    override fun rename(name: String): ONNXData<NDArray, WebGPUBackend> = WebGPUTensor(name, data, info)

    companion object {
        @Suppress("UNCHECKED_CAST")
        fun create(proto: TensorProto, device: Device): WebGPUTensor {
            val type = proto.dataType ?: TensorProto.DataType.UNDEFINED
            val array = parseArray(proto)
            requireNotNull(array) { "Array value should be initialized" }

            return WebGPUTensor(array, type, proto.dims, proto.name, device)
        }

        private operator fun invoke(value: Any, type: TensorProto.DataType, shape: IntArray, name: String? = "", device: Device): WebGPUTensor {
            val nameNotNull = name.orEmpty()
            return when (type) {
                TensorProto.DataType.INT8 -> NDArray(ArrayInfo(shape, WebGPUDataType.INT32), (value as ByteArray).toIntArray(), device = device).asTensor(nameNotNull)
                TensorProto.DataType.INT16 -> NDArray(ArrayInfo(shape, WebGPUDataType.INT32), (value as ShortArray).toIntArray(), device = device).asTensor(nameNotNull)
                TensorProto.DataType.INT32 -> NDArray(ArrayInfo(shape, WebGPUDataType.INT32), value as IntArray, device = device).asTensor(nameNotNull)
                TensorProto.DataType.INT64 -> NDArray(ArrayInfo(shape, WebGPUDataType.INT32), (value as LongArray).toIntArray(), device = device).asTensor(nameNotNull)
                TensorProto.DataType.UINT8 -> NDArray(ArrayInfo(shape, WebGPUDataType.UINT32), (value as UByteArray).toUIntArray().asIntArray(), device = device).asTensor(nameNotNull)
                TensorProto.DataType.UINT16 -> NDArray(ArrayInfo(shape, WebGPUDataType.UINT32), (value as UShortArray).toUIntArray().asIntArray(), device = device).asTensor(nameNotNull)
                TensorProto.DataType.UINT32 -> NDArray(ArrayInfo(shape, WebGPUDataType.UINT32), (value as UIntArray).asIntArray(), device = device).asTensor(nameNotNull)
                TensorProto.DataType.UINT64 -> NDArray(ArrayInfo(shape, WebGPUDataType.UINT32), (value as ULongArray).toUIntArray().asIntArray(), device = device).asTensor(nameNotNull)
                TensorProto.DataType.FLOAT -> NDArray(ArrayInfo(shape, WebGPUDataType.FLOAT32), value as FloatArray, device = device).asTensor(nameNotNull)
                TensorProto.DataType.DOUBLE -> NDArray(ArrayInfo(shape, WebGPUDataType.FLOAT32), (value as DoubleArray).toFloatArray(), device = device).asTensor(nameNotNull)
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
