package io.kinference.tfjs.data.tensors

import io.kinference.data.ONNXTensor
import io.kinference.ndarray.arrays.*
import io.kinference.protobuf.FLOAT_TENSOR_TYPES
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.tfjs.TFJSBackend
import io.kinference.types.ValueInfo
import io.kinference.types.ValueTypeInfo
import io.kinference.utils.ArrayUsageMarker

class TFJSTensor(name: String?, override val data: NDArrayTFJS, val info: ValueTypeInfo.TensorTypeInfo) : ONNXTensor<NDArrayTFJS, TFJSBackend>(name, data) {
    constructor(data: NDArrayTFJS, info: ValueInfo) : this(info.name, data, info.typeInfo as ValueTypeInfo.TensorTypeInfo)

    override val backend = TFJSBackend

    override fun rename(name: String): TFJSTensor {
        return TFJSTensor(name, data, info)
    }

    override fun close() {
        data.close()
    }

    override fun clone(newName: String?): TFJSTensor {
        return TFJSTensor(newName, data.clone(), info)
    }

    override fun markOutput(marker: ArrayUsageMarker) {
//        TODO("Not yet implemented")
    }

    companion object {
        //TODO: complex, uint32/64 tensors
        fun create(proto: TensorProto): TFJSTensor {
            val type = proto.dataType ?: DataType.UNDEFINED
            val array = parseArray(proto)

            return TFJSTensor(array, type, proto.dims, proto.name)
        }

        private operator fun invoke(value: Any, type: DataType, dims: IntArray, name: String? = ""): TFJSTensor {
            val nameNotNull = name.orEmpty()
            val typedDims = dims.toTypedArray()
            return when (type) {
                in FLOAT_TENSOR_TYPES -> NDArrayTFJS.float(value as FloatArray, typedDims)
                DataType.DOUBLE -> NDArrayTFJS.float(value as DoubleArray, typedDims)
                DataType.INT8 -> NDArrayTFJS.int(value as ByteArray, typedDims)
                DataType.INT16 -> NDArrayTFJS.int(value as ShortArray, typedDims)
                DataType.INT32 -> NDArrayTFJS.int(value as IntArray, typedDims)
                DataType.INT64 -> NDArrayTFJS.int(value as LongArray, typedDims)
                DataType.UINT8 -> NDArrayTFJS.int(value as UByteArray, typedDims)
                DataType.UINT16 -> NDArrayTFJS.int(value as UShortArray, typedDims)
                DataType.UINT32 -> NDArrayTFJS.int(value as UIntArray, typedDims)
                DataType.UINT64 -> NDArrayTFJS.int(value as ULongArray, typedDims)
                DataType.BOOL -> NDArrayTFJS.boolean(value as BooleanArray, typedDims)
                DataType.STRING -> NDArrayTFJS.string(value as Array<String>, typedDims)
                else -> error("Unsupported type: $type")
            }.asTensor(nameNotNull)
        }

        private fun parseArray(proto: TensorProto): Any {
            val array = if (proto.isString()) {
                val data = proto.stringData
                Array(data.size) { data[it].utf8() }
            } else {
                proto.arrayData
            }
            requireNotNull(array) { "Array value should be initialized" }
            return array
        }
    }
}
