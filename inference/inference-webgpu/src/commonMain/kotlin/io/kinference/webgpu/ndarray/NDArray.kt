package io.kinference.webgpu.ndarray

import io.kinference.utils.webgpu.*
import io.kinference.webgpu.graph.WebGPUState

class NDArray private constructor(val info: NDArrayInfo, private var state: NDArrayState) {
    fun getBuffer(gpuState: WebGPUState): Buffer {
        prepareBuffer(gpuState)
        return (state as NDArrayBuffer).buffer
    }

    suspend fun finalizeOutputNDArray(gpuState: WebGPUState) {
        getData(gpuState)
        when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData -> currentState.data
            is NDArrayInitializedBuffer -> {
                state = NDArrayData(currentState.data)
            }
            is NDArrayUninitializedBuffer, is NDArrayCopyingBuffer -> error("Incorrect state")
        }
    }

    fun requestData(gpuState: WebGPUState): Unit =
        when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData, is NDArrayInitializedBuffer, is NDArrayCopyingBuffer -> {}
            is NDArrayUninitializedBuffer -> {
                state = NDArrayCopyingBuffer(
                    sourceBuffer = currentState.buffer,
                    destinationBuffer = gpuState.createReadableBuffer(info, currentState.buffer)
                )
            }
        }

    fun getData(): TypedNDArrayData = (state as NDArrayData).data

    suspend fun getData(gpuState: WebGPUState): TypedNDArrayData =
        when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData -> currentState.data
            is NDArrayInitializedBuffer -> currentState.data
            is NDArrayUninitializedBuffer -> {
                requestData(gpuState)
                getData(gpuState)
            }
            is NDArrayCopyingBuffer -> {
                gpuState.enqueueCommands()
                currentState.destinationBuffer.mapAsync(MapModeFlags(MapMode.Read))
                val data = currentState.destinationBuffer.getMappedRange().unpack(info)
                state = NDArrayInitializedBuffer(data, currentState.sourceBuffer)
                data
            }
        }

    private fun prepareBuffer(gpuState: WebGPUState): Unit =
        when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData -> {
                state = NDArrayInitializedBuffer(currentState.data, gpuState.createInitializedBuffer(info, currentState.data))
            }
            is NDArrayBuffer -> {}
        }

    fun destroy() {
        val currentState = state
        if (currentState is NDArrayBuffer) {
            currentState.buffer.destroy()
        }
        state = NDArrayDestroyed
    }

    fun reshape(newShape: IntArray, gpuState: WebGPUState): NDArray {
        val newInfo = NDArrayInfo(newShape, info.type)
        return when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData -> NDArray(newInfo, NDArrayData(currentState.data))
            is NDArrayInitializedBuffer -> NDArray(newInfo, NDArrayData(currentState.data))
            is NDArrayBuffer -> {
                val newBuffer = gpuState.createUninitializedBuffer(newInfo)
                gpuState.copyBufferToBuffer(currentState.buffer, 0, newBuffer, 0, newInfo.size)
                NDArray(newInfo, NDArrayUninitializedBuffer(newBuffer))
            }
        }
    }

    companion object {
        fun intNDArray(info: NDArrayInfo, data: IntArray): NDArray = NDArray(info, IntNDArrayData(data))
        fun uintNDArray(info: NDArrayInfo, data: UIntArray): NDArray = NDArray(info, UIntNDArrayData(data))
        fun floatNDArray(info: NDArrayInfo, data: FloatArray): NDArray = NDArray(info, FloatNDArrayData(data))

        fun scalar(value: Int) = intNDArray(NDArrayInfo(intArrayOf(), WebGPUDataType.INT32), intArrayOf(value))
        fun scalar(value: UInt) = uintNDArray(NDArrayInfo(intArrayOf(), WebGPUDataType.UINT32), uintArrayOf(value))
        fun scalar(value: Float) = floatNDArray(NDArrayInfo(intArrayOf(), WebGPUDataType.FLOAT32), floatArrayOf(value))

        operator fun invoke(info: NDArrayInfo, data: TypedNDArrayData): NDArray = NDArray(info, NDArrayData(data))

        operator fun invoke(info: NDArrayInfo, gpuState: WebGPUState): NDArray =
            NDArray(info, NDArrayUninitializedBuffer(gpuState.createUninitializedBuffer(info)))
    }
}
