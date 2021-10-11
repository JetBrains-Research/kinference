package io.kinference

import io.kinference.utils.*

actual object TestLoggerFactory {
    actual fun create(name: String): KILogger {
        val logger = log.getLogger(name)
        logger.setDefaultLevel("INFO")
        return LogLevelLogger(logger)
    }
}
