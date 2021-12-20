package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.*
import io.kinference.utils.wgpu.jnr.*
import io.kinference.utils.wgpu.jnr.utils.libraryFileName
import jnr.ffi.LibraryLoader
import jnr.ffi.Runtime
import java.io.File
import kotlin.io.path.createFile
import kotlin.io.path.outputStream

actual object WebGPUInstance {
    val wgpuNative: WgpuNative

    init {
        val sharedLibrary = loadWgpuNative().absolutePath
        wgpuNative = LibraryLoader.create(WgpuNative::class.java).load(sharedLibrary)
        WgpuRuntime.runtime = Runtime.getRuntime(wgpuNative)
    }

    actual suspend fun requestAdapter(options: RequestAdapterOptions): Adapter {
        var wgpuAdapter: WGPUAdapter? = null
        var wgpuError: Exception? = null
        var wgpuStatus: WGPURequestAdapterStatus? = null

        wgpuNative.wgpuInstanceRequestAdapter(
            nullptr,
            options.getPointerTo(),
            { status, adapter, message, _ ->
                wgpuStatus = status
                if (status == WGPURequestAdapterStatus.Success) {
                    wgpuAdapter = adapter
                } else if (!message.isNullptr) {
                    wgpuError = RuntimeException(message.getString())
                }
            },
            nullptr
        )
        return wgpuAdapter?.let { Adapter(it) }
            ?: throw wgpuError ?: error("requestAdapter() failed: status $wgpuStatus")
    }

    private fun loadWgpuNative(): File {
        // FIXME file is copied each time
        val libraryName = libraryFileName("wgpu")
        val inputStream = WgpuNative::class.java.getResourceAsStream("/$libraryName")
            ?: error("wgpu library not found")

        val tempDirectory = kotlin.io.path.createTempDirectory("wgpu")

        val libraryFile = tempDirectory.resolve(libraryName).createFile()
        libraryFile.outputStream().apply {
            inputStream.copyTo(this)
            close()
        }
        inputStream.close()

        return libraryFile.toFile()
    }
}
