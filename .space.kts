val jsContainer = "registry.jetbrains.team/p/ki/containers-ci/ci-corretto-17-firefox:1.0.0"
val jvmContainer = "amazoncorretto:17"

job("KInference / Build and Test") {
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
            cacheTestData()

            kotlinScript { api ->
                api.gradlew("jvmHeavyTest", "--console=plain", "-Pci", "--no-daemon")
            }
        }

        container("JS Legacy Heavy Tests", jsContainer) {
            addAwsKeys()
            cacheTestData()

            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsLegacyHeavyTest --console=plain -Pci --no-daemon"
            }
        }

        container("JS IR Heavy Tests", jsContainer) {
            addAwsKeys()
            cacheTestData()

            shellScript {
                content = "xvfb-run --auto-servernum ./gradlew jsIrHeavyTest --console=plain -Pci --no-daemon"
            }
        }
    }
}

job("KInference / Release") {
    startOn {
        gitPush {
            enabled = false
        }
    }

    container("Release",jvmContainer) {
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

fun Container.cacheTestData() {
    cache {
        storeKey = "test-data-{{ hashFiles('buildSrc/src/main/kotlin/io/kinference/gradle/s3/DefaultS3Deps.kt') }}"
        localPath = "test-data"
    }
}
