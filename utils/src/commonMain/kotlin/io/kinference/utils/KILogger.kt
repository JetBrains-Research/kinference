package io.kinference.utils

interface KILogger {
    fun debug(message: () -> String)
    fun info(message: () -> String)
    fun warning(message: () -> String)
    fun error(message: () -> String, e: Throwable? = null)
}

expect object LoggerFactory {
    fun create(name: String): KILogger
}
