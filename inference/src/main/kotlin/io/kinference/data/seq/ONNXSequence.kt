package io.kinference.data.seq

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.types.ValueInfo

class ONNXSequence(val data: List<ONNXData>, info: ValueInfo) : ONNXData(ONNXDataType.ONNX_SEQUENCE, info) {
    constructor(info: ValueInfo, size: Int, init: (Int) -> ONNXData) : this(List(size, init), info)

    override fun rename(name: String): ONNXData = ONNXSequence(data, ValueInfo(info.typeInfo, name))

    val length: Int = data.size
}
