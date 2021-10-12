package io.kinference.utils

@JsNonModule
@JsModule("loglevel")
external val log: RootLogger

external interface LogLevel {
    var TRACE: Number /* 0 */
    var DEBUG: Number /* 1 */
    var INFO: Number /* 2 */
    var WARN: Number /* 3 */
    var ERROR: Number /* 4 */
    var SILENT: Number /* 5 */
}

typealias LoggingMethod = (message: Any) -> Unit

typealias MethodFactory = (methodName: String, level: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, loggerName: dynamic /* String | Any */) -> LoggingMethod

external interface RootLogger : Logger {
    fun noConflict(): Any
    fun getLogger(name: String): Logger
    fun getLogger(name: Any): Logger
    var default: RootLogger
}

external interface Logger {
    val levels: LogLevel
    var methodFactory: MethodFactory
    fun trace(vararg msg: Any)
    fun debug(vararg msg: Any)
    fun log(vararg msg: Any)
    fun info(vararg msg: Any)
    fun warn(vararg msg: Any)
    fun error(vararg msg: Any)
    fun setLevel(level: Any /* 0 | 1 | 2 | 3 | 4 | 5 | "trace" | "debug" | "info" | "warn" | "error" | "silent" | "TRACE" | "DEBUG" | "INFO" | "WARN" | "ERROR" | "SILENT" */, persist: Boolean = definedExternally)
    fun getLevel(): Number /* 0 | 1 | 2 | 3 | 4 | 5 */
    fun setDefaultLevel(level: Any /* 0 | 1 | 2 | 3 | 4 | 5 | "trace" | "debug" | "info" | "warn" | "error" | "silent" | "TRACE" | "DEBUG" | "INFO" | "WARN" | "ERROR" | "SILENT" */)
    fun enableAll(persist: Boolean = definedExternally)
    fun disableAll(persist: Boolean = definedExternally)
}
