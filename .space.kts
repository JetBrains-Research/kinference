val jsContainer = "registry.jetbrains.team/p/ki/containers-ci/ci-corretto-17-firefox:1.0.0"
val jvmContainer = "amazoncorretto:17"

/*job("KInference / Build and Test") {
    parallel {
        container("Build With Gradle",jvmContainer) {
            kotlinScript { api ->
                api.gradlew("assemble", "--parallel", "--console=plain", "--no-daemon")
            }
        }

        container("JVM Tests", jvmContainer) {
            kotlinScript { api ->
                api.gradlew("jvmTest", "--parallel", "--console=plain", "-Pci", "--no-daemon")
            }
        }

        container("JS Legacy Tests", jsContainer) {
            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsLegacyTest --parallel --console=plain -Pci --no-daemon"
            }
        }

        container("JS IR Tests", jsContainer) {
            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsIrTest jsTest --parallel --console=plain -Pci --no-daemon"
            }
        }

        container("JVM Heavy Tests", jvmContainer) {
            addAwsKeys()

            kotlinScript { api ->
                api.gradlew("jvmHeavyTest", "--console=plain", "-Pci", "--no-daemon")
            }
        }

        container("JS Legacy Heavy Tests", jsContainer) {
            addAwsKeys()

            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsLegacyHeavyTest --console=plain -Pci --no-daemon"
            }
        }

        container("JS IR Heavy Tests", jsContainer) {
            addAwsKeys()

            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsIrHeavyTest --console=plain -Pci --no-daemon"
            }
        }
    }
}*/

job("KInference / Build and Test") {
    container("Build With Gradle", jvmContainer) {
        gradleCache()

        kotlinScript { api ->
            api.gradlew("assemble", "--parallel", "--console=plain", "--no-daemon")
        }
    }
}

job("KInference / JVM Test") {
    container("JVM Tests", jvmContainer) {
        gradleCache()

        kotlinScript { api ->
            api.gradlew("jvmTest", "--parallel", "--console=plain", "-Pci", "--no-daemon")
        }
    }
}

job("KInference / JS IR Test") {
    container("JS IR Tests", jsContainer) {
        gradleCache()

        shellScript {
            content = xvfbRun("./gradlew jsIrTest jsTest --parallel --console=plain -Pci --no-daemon")
        }
    }
}

job("KInference / JS Legacy Test") {
    container("JS Legacy Tests", jsContainer) {
        gradleCache()

        shellScript {
            content = xvfbRun("./gradlew jsLegacyTest --parallel --console=plain -Pci --no-daemon")
        }
    }
}

job("KInference / JVM Heavy Test") {
    container("JVM Heavy Tests", jvmContainer) {
        gradleCache()

        addAwsKeys()

        kotlinScript { api ->
            api.gradlew("jvmHeavyTest", "--console=plain", "-Pci", "--no-daemon")
        }
    }
}

job("KInference / JS IR Heavy Test") {
    container("JS IR Heavy Tests", jsContainer) {
        gradleCache()
        addAwsKeys()

        shellScript {
            content = xvfbRun("./gradlew jsIrHeavyTest -x :inference:inference-core:jsIrHeavyTest --console=plain -Pci --no-daemon")
        }
    }
}

job("KInference / JS Legacy Heavy Test") {
    container("JS Legacy Heavy Tests", jsContainer) {
        gradleCache()
        addAwsKeys()

        shellScript {
            content = xvfbRun("./gradlew jsLegacyHeavyTest -x :inference:inference-core:jsLegacyHeavyTest --console=plain -Pci --no-daemon")
        }
    }
}

job("KInference / Inference Core / JS Legacy Heavy Test ") {
    container("JS Legacy Heavy Tests", jsContainer) {
        gradleCache()
        addAwsKeys()

        shellScript {
            content = xvfbRun("./gradlew :inference:inference-core:jsLegacyHeavyTest --console=plain -Pci --no-daemon")
        }
    }
}

job("KInference / Inference Core / JS IR Heavy Test ") {
    container("JS IR Heavy Tests", jsContainer) {
        gradleCache()
        addAwsKeys()

        shellScript {
            content = xvfbRun("./gradlew :inference:inference-core:jsIrHeavyTest --console=plain -Pci --no-daemon")
        }
    }
}

job("KInference / Release") {
    startOn {
        gitPush {
            enabled = false
        }
    }

    container("Release", jvmContainer) {
        addAwsKeys()

        kotlinScript { api ->
            api.gradlew("publish", "--parallel", "--console=plain", "--no-daemon")
        }
    }
}


fun Container.addAwsKeys() {
    env["AWS_ACCESS_KEY"] = "{{ project:aws_access_key }}"
    env["AWS_SECRET_KEY"] = "{{ project:aws_secret_key }}"
}

fun xvfbRun(command: String): String = "xvfb-run --auto-servernum $command"

fun Container.gradleCache() {
    env["GRADLE_USER_HOME"] = "~/.gradle/"

    cache {
        this.localPath = "~/.gradle/caches"

        this.storeKey = "gradle-cache-{{ run:job.id }}-{{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties', 'buildSrc/**/Versions.kt') }}"

        this.restoreKeys {
            +"gradle-cache-master"
        }
    }

    cache {
        this.localPath = "~/.gradle/wrapper"

        this.storeKey = "gradle-wrapper-{{ run:job.id }}-{{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties', 'buildSrc/**/Versions.kt') }}"

        this.restoreKeys {
            +"gradle-wrapper-master"
        }
    }
}
