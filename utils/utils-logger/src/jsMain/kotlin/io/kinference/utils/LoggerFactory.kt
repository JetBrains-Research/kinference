package io.kinference.utils

class LogLevelLogger(private val logger: Logger): KILogger {
    override fun debug(message: () -> String) {
        logger.debug(message())
        logger.debug("\n")
    }

    override fun info(message: () -> String) {
        logger.info(message())
        logger.info("\n")
    }

    override fun warning(message: () -> String) {
        logger.warn(message())
        logger.warn("\n")
    }

    override fun error(message: () -> String, e: Throwable?) {
        if (e == null) {
            logger.error(message())
        } else {
            logger.error(message() + "\n" + e.stackTraceToString())
        }
        logger.error("\n")
    }
}

actual object LoggerFactory {
    actual fun create(name: String): KILogger {
        return LogLevelLogger(log.getLogger(name))
    }
}
