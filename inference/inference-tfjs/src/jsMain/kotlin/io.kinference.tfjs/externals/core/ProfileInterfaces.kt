package io.kinference.tfjs.externals.core

import io.kinference.ndarray.arrays.ArrayTFJS
import kotlin.js.Promise

interface ProfileInfo {
    val newBytes: Int
    val newTensors: Int
    val peakBytes: Int
    val kernels: Array<KernelInfo>
    val result: Array<ArrayTFJS>
    val kernelNames: Array<String>
}

interface KernelInfo {
    val name: String
    val bytesAdded: Int
    val totalBytesSnapshot: Int
    val tensorsAdded: Int
    val totalTensorsSnapshot: Int
    val inputShapes: Array<Array<Int>>
    val outputShapes: Array<Array<Int>>
    val kernelTimeMs: Int
    val extraInfo: Promise<String>
}

interface TimingInfo {
    val kernelMs: Int
    val wallMs: Int
    fun getExtraProfileInfo(): String?
}

interface MemoryInfo {
    val numTensors: Int
    val numDataBuffers: Int
    val numBytes: Int
    val unreliable: Boolean?
    val reasons: Array<String>
}
