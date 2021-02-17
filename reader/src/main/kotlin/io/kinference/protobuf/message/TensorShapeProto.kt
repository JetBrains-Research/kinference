package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TensorShapeProto(
    //ProtoTag = 1
    val dim: List<Dimension> = emptyList(),
) {
    companion object {
        fun decode(reader: ProtobufReader): TensorShapeProto {
            val dim = mutableListOf<Dimension>()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> dim.add(Dimension.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return TensorShapeProto(dim = dim)
        }
    }

    class Dimension(
        //ProtoTag = 3
        val denotation: String? = null,

        //ProtoTag = 1
        val dim_value: Long? = null,

        //ProtoTag = 2
        val dim_param: String? = null
    ) {
        companion object {
            fun decode(reader: ProtobufReader): Dimension {
                var denotation: String? = null
                var dim_value: Long? = null
                var dim_param: String? = null
                reader.forEachTag { tag ->
                    when (tag) {
                        3 -> denotation = reader.readString()
                        1 -> dim_value = reader.readLong()
                        2 -> dim_param = reader.readString()
                        else -> reader.readUnknownField(tag)
                    }
                }
                return Dimension(denotation = denotation, dim_value = dim_value, dim_param = dim_param)
            }
        }
    }
}
