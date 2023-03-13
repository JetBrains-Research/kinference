package io.kinference.tfjs.data.tensors

import io.kinference.data.ONNXTensor
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.tensor
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.protobuf.toIntArray
import io.kinference.tfjs.TFJSBackend
import io.kinference.types.ValueInfo
import io.kinference.types.ValueTypeInfo

class TFJSTensor(name: String?, override val data: NDArrayTFJS, val info: ValueTypeInfo.TensorTypeInfo) : ONNXTensor<NDArrayTFJS, TFJSBackend>(name, data) {
    constructor(data: NDArrayTFJS, info: ValueInfo) : this(info.name, data, info.typeInfo as ValueTypeInfo.TensorTypeInfo)

    override val backend = TFJSBackend

    override fun rename(name: String): TFJSTensor {
        return TFJSTensor(name, data, info)
    }

    override fun close() {
        data.close()
    }

    /*fun toNDArray(): NDArray {
        val shapeIntArray = data.shape.toIntArray()
        val strides = Strides(shapeIntArray)
        val blockSize = blockSizeByStrides(strides)
        val blocksCount = strides.linearSize / blockSize


        return when(data.dtype) {
            "float32" -> {
                val array = data.dataFloat().unsafeCast<Float32Array>()
                val arrayBuffer = array.buffer
                val blocks = Array(blocksCount) { blockNum ->
                    Float32Array(arrayBuffer, blockNum * blockSize * 4, blockSize).unsafeCast<FloatArray>()
                }
                val tiledArray = FloatTiledArray(blocks)
                FloatNDArray(tiledArray, Strides(shapeIntArray))
            }

            "int32" -> {
                val array = data.dataFloat().unsafeCast<Int32Array>()
                val arrayBuffer = array.buffer
                val blocks = Array(blocksCount) { blockNum ->
                    Int32Array(arrayBuffer, blockNum * blockSize * 4, blockSize).unsafeCast<IntArray>()
                }
                val tiledArray = IntTiledArray(blocks)
                IntNDArray(tiledArray, strides)
            }

            "bool" -> {
                val array = data.dataBool()
                val tiledArray = BooleanTiledArray(shapeIntArray) { array[it] }
                BooleanNDArray(tiledArray, strides)
            }

            else -> error("Unsupported type")
        }
    }*/

    companion object {
        //TODO: complex, uint32/64 tensors
        fun create(proto: TensorProto): TFJSTensor {
            val type = proto.dataType ?: DataType.UNDEFINED
            val array = parseArray(proto)

            return TFJSTensor(array, type, proto.dims, proto.name)
        }

        /*operator fun invoke(value: NDArray, name: String? = ""): TFJSTensor {
            return when (val resolvedType = value.type.resolveProtoDataType()) {
                DataType.FLOAT -> invoke((value as FloatNDArray).array.toArray(), resolvedType, value.shape, name)
                DataType.INT32 -> invoke((value as IntNDArray).array.toArray(), resolvedType, value.shape, name)
                DataType.UINT8 -> invoke((value as UByteNDArray).array.toArray(), resolvedType, value.shape, name)
                DataType.INT64 -> invoke((value as LongNDArray).array.toArray(), resolvedType, value.shape, name)
                DataType.BOOL  -> invoke((value as BooleanNDArray).array.toArray(), resolvedType, value.shape, name)
                else -> error("Unsupported type")
            }
        }*/

        private operator fun invoke(value: Any, type: DataType, dims: IntArray, name: String? = ""): TFJSTensor {
            val nameNotNull = name.orEmpty()
            val typedDims = dims.toTypedArray()
            return when (type) {
                DataType.FLOAT -> NumberNDArrayTFJS(tensor(value as FloatArray, typedDims, "float32")).asTensor(nameNotNull)
                DataType.INT32 -> NumberNDArrayTFJS(tensor(value as IntArray, typedDims, "int32")).asTensor(nameNotNull)
                DataType.UINT8 -> NumberNDArrayTFJS(tensor((value as UByteArray).toTypedArray(), typedDims, "int32")).asTensor(nameNotNull)
                DataType.INT8  -> NumberNDArrayTFJS(tensor((value as ByteArray).toTypedArray(), typedDims, "int32")).asTensor(nameNotNull)
                DataType.INT64 -> NumberNDArrayTFJS(tensor((value as LongArray).toIntArray(), typedDims, "int32")).asTensor(nameNotNull)
                DataType.BOOL -> BooleanNDArrayTFJS(tensor((value as BooleanArray).toTypedArray(), typedDims, "bool")).asTensor(nameNotNull)
                else -> error("Unsupported type: $type")
            }
        }

        private fun parseArray(proto: TensorProto): Any {
            val array = if (proto.isString()) proto.stringData else proto.arrayData
            requireNotNull(array) { "Array value should be initialized" }
            return array
        }
    }
}
