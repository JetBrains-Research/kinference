package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.LongArrayDeserializer

class SparseTensorProto(
    //ProtoTag = 1
    val values: TensorProto? = null,

    //ProtoTag = 2
    val indices: TensorProto? = null,

    //ProtoTag = 3
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
                    3 -> dims = LongArrayDeserializer.decode(reader, tag)
                    else -> reader.readUnknownField(tag)
                }
            }
            return SparseTensorProto(values = values, indices = indices, dims = dims)
        }
    }
}
