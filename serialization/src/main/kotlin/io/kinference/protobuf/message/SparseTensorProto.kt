package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.LongArraySerializer

class SparseTensorProto(
    val values: TensorProto? = null,
    val indices: TensorProto? = null,
    val dims: LongArray? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): SparseTensorProto {
            var values: TensorProto? = null
            var indices: TensorProto? = null
            var dims: LongArray? = null
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> values = TensorProto.decode(reader)
                    2 -> indices = TensorProto.decode(reader)
                    3 -> dims = LongArraySerializer.decode(reader, tag)
                    else -> reader.readUnknownField(tag)
                }
            }
            return SparseTensorProto(values = values, indices = indices, dims = dims)
        }
    }
}
