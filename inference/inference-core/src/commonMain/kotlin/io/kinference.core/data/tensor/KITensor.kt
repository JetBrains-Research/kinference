package io.kinference.core.data.tensor

import io.kinference.core.CoreBackend
import io.kinference.data.ONNXTensor
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.protobuf.FLOAT_TENSOR_TYPES
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.types.ValueInfo
import io.kinference.types.ValueTypeInfo

//TODO: support segments
//TODO: support external data
class KITensor(name: String?, override val data: NDArrayCore, val info: ValueTypeInfo.TensorTypeInfo) : ONNXTensor<NDArrayCore, CoreBackend>(name, data) {
    constructor(data: NDArrayCore, info: ValueInfo) : this(info.name, data, info.typeInfo as ValueTypeInfo.TensorTypeInfo)

    override fun close() {
        data.close()
    }

    suspend operator fun minus(other: KITensor): KITensor {
        require(this.data is NumberNDArrayCore && other.data is NumberNDArrayCore)
        return (this.data - other.data).asTensor()
    }

    suspend operator fun times(other: KITensor): KITensor {
        require(this.data is NumberNDArrayCore && other.data is NumberNDArrayCore)
        return (this.data * other.data).asTensor()
    }

    suspend operator fun div(other: KITensor): KITensor {
        require(this.data is NumberNDArrayCore && other.data is NumberNDArrayCore)
        return (this.data / other.data).asTensor()
    }

    override val backend = CoreBackend

    override fun rename(name: String): KITensor {
        return KITensor(name, data, info)
    }

    companion object {
        //TODO: complex, uint32/64 tensors
        fun create(proto: TensorProto): KITensor {
            val type = proto.dataType ?: DataType.UNDEFINED
            val array = parseArray(proto)

            return KITensor(array, type, proto.dims, proto.name)
        }

        private operator fun invoke(value: Any, type: DataType, dims: IntArray = IntArray(0), name: String? = ""): KITensor {
            val name = name ?: ""
            val strides = Strides(dims)
            return when (type) {
                DataType.DOUBLE -> DoubleNDArray(value as DoubleTiledArray, strides).asTensor(name)
                in FLOAT_TENSOR_TYPES -> FloatNDArray(value as FloatTiledArray, strides).asTensor(name)
                DataType.INT8 -> ByteNDArray(value as ByteTiledArray, strides).asTensor(name)
                DataType.INT16 -> ShortNDArray(value as ShortTiledArray, strides).asTensor(name)
                DataType.INT32 -> IntNDArray(value as IntTiledArray, strides).asTensor(name)
                DataType.INT64 -> LongNDArray(value as LongTiledArray, strides).asTensor(name)
                DataType.UINT8 -> UByteNDArray(value as UByteTiledArray, strides).asTensor(name)
                DataType.UINT16 -> UShortNDArray(value as UShortTiledArray, strides).asTensor(name)
                DataType.UINT32 -> UIntNDArray(value as UIntTiledArray, strides).asTensor(name)
                DataType.UINT64 -> ULongNDArray(value as ULongTiledArray, strides).asTensor(name)
                DataType.BOOL -> BooleanNDArray(value as BooleanTiledArray, strides).asTensor(name)
                DataType.STRING -> {
                    value as List<String>
                    StringNDArray(dims) { value[it] }.asTensor(name)
                }
                else -> error("Unsupported data type $type")
            }
        }

        private fun parseArray(proto: TensorProto): Any {
            val array = if (proto.isString()) proto.stringData else proto.arrayData
            requireNotNull(array) { "Array value should be initialized" }
            return array
        }
    }
}
