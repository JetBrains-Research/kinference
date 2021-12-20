package io.kinference.utils.wgpu.jnr.utils

fun libraryFileName(libraryName: String): String {
    val isWindows = System.getProperty("os.name").contains("Windows")
    val isLinux = System.getProperty("os.name").contains("Linux")
    val isMac = System.getProperty("os.name").contains("Mac")

    return when {
        isWindows -> "$libraryName.dll"
        isLinux -> "lib$libraryName.so"
        isMac -> "lib$libraryName.dylib"
        else -> error("Unsupported platform")
    }
}
