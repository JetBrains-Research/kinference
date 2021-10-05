package io.kinference.data.tensors

import io.kinference.custom_externals.core.*
import io.kinference.custom_externals.extensions.tensor
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.arrays.pointers.forEach
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.types.ValueInfo

class Tensor(val data: TensorTFJS, info: ValueInfo) : ONNXData(ONNXDataType.ONNX_TENSOR, info) {

    override fun rename(name: String): Tensor {
        return Tensor(data, ValueInfo(info.typeInfo, name))
    }

    companion object {
        //TODO: complex, uint32/64 tensors
        @Suppress("UNCHECKED_CAST")
        fun create(proto: TensorProto): Tensor {
            val type = proto.dataType ?: DataType.UNDEFINED
            val array = parseArray(proto)
            requireNotNull(array) { "Array value should be initialized" }

            return Tensor(array, type, proto.dims, proto.name)
        }

        private operator fun invoke(value: Any, type: DataType, dims: IntArray, name: String? = ""): Tensor {
            val nameNotNull = name.orEmpty()
//            val actualDims = if (dims.isEmpty()) intArrayOf(1) else dims
            val typedDims = dims.toTypedArray()
            return when (type) {
                DataType.FLOAT -> tensor((value as FloatTiledArray).toArray(), typedDims, "float32").asTensor(nameNotNull)
                DataType.INT32 -> tensor((value as IntTiledArray).toArray(), typedDims, "int32").asTensor(nameNotNull)
                DataType.UINT8 -> tensor((value as UByteTiledArray).toArray(), typedDims, "int32").asTensor(nameNotNull)
                DataType.INT64 -> {
                    value as LongTiledArray
                    val outputIntArray = IntArray(value.size)
                    var count = 0
                    value.pointer().forEach(value.size) {
                        outputIntArray[count++] = it.toInt()
                    }
                    tensor(outputIntArray, typedDims, dtype = "int32").asTensor(nameNotNull)
                }
                else -> error("Unsupported type")
            }
        }

        private fun parseArray(proto: TensorProto) = when {
            proto.isTiled() -> proto.tiledData
            proto.isString() -> proto.stringData
            else -> error("Unsupported data type ${proto.dataType}")
        }
    }
}
