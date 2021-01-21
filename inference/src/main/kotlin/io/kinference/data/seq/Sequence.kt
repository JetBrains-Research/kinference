package io.kinference.data.seq

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.types.ValueInfo

class Sequence(val data: List<ONNXData>, override val info: ValueInfo.SequenceInfo) : ONNXData(ONNXDataType.ONNX_SEQUENCE, info) {
    override fun rename(name: String): ONNXData = Sequence(data, ValueInfo.SequenceInfo(name, info.type))

    val length: Int = data.size
}
