package io.kinference.model

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.message.ModelProto
import okio.Buffer
import java.io.File

class Model(proto: ModelProto) {
    val graph = Graph(proto.graph!!)

    fun predict(input: Collection<ONNXData>): List<ONNXData> {
        return graph.execute(input.toList())
    }

    companion object {
        fun load(file: String): Model = load(File(file).readBytes())

        fun load(bytes: ByteArray): Model {
            val buffer = Buffer().write(bytes)
            val modelScheme = ModelProto.decode(ProtobufReader(buffer))
            return Model(modelScheme)
        }
    }
}
