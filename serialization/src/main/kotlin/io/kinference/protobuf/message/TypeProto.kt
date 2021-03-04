package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TypeProto(
    val denotation: String? = null,
    val tensorType: Tensor? = null,
    val sequenceType: Sequence? = null,
    val mapType: Map? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): TypeProto {
            var denotation: String? = null
            var tensorType: Tensor? = null
            var sequenceType: Sequence? = null
            var mapType: Map? = null
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> tensorType = Tensor.decode(reader)
                    4 -> sequenceType = Sequence.decode(reader)
                    5 -> mapType = Map.decode(reader)
                    6 -> denotation = reader.readString()
                    else -> reader.readUnknownField(tag)
                }
            }
            return TypeProto(
                denotation = denotation,
                tensorType = tensorType,
                sequenceType = sequenceType,
                mapType = mapType
            )
        }
    }

    class Tensor(
        //ProtoTag = 1
        val elem_type: Int? = null,

        //ProtoTag = 2
        val shape: TensorShapeProto? = null
    ) {
        companion object {
            fun decode(reader: ProtobufReader): Tensor {
                var elem_type: Int? = null
                var shape: TensorShapeProto? = null
                reader.forEachTag { tag ->
                    when (tag) {
                        1 -> elem_type = reader.readInt()
                        2 -> shape = TensorShapeProto.decode(reader)
                        else -> reader.readUnknownField(tag)
                    }
                }
                return Tensor(elem_type = elem_type, shape = shape)
            }
        }
    }

    class Sequence(
        //ProtoTag = 1
        val elem_type: TypeProto? = null
    ) {
        companion object {
            fun decode(reader: ProtobufReader): Sequence {
                var elem_type: TypeProto? = null
                reader.forEachTag { tag ->
                    when (tag) {
                        1 -> elem_type = TypeProto.decode(reader)
                        else -> reader.readUnknownField(tag)
                    }
                }
                return Sequence(elem_type = elem_type)
            }
        }
    }

    class Map(
        //ProtoTag = 1
        val key_type: Int? = null,

        //ProtoTag = 2
        val value_type: TypeProto? = null
    ) {
        companion object {
            fun decode(reader: ProtobufReader): Map {
                var key_type: Int? = null
                var value_type: TypeProto? = null
                reader.forEachTag { tag ->
                    when (tag) {
                        1 -> key_type = reader.readInt()
                        2 -> value_type = TypeProto.decode(reader)
                        else -> reader.readUnknownField(tag)
                    }
                }
                return Map(key_type = key_type, value_type = value_type)
            }
        }
    }
}
