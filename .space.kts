val jsContainer = "registry.jetbrains.team/p/ki/containers-ci/ci-corretto-17-firefox:1.0.1"
val jvmContainer = "amazoncorretto:17"

job("KInference / Build") {
    container("Build With Gradle", jvmContainer) {
        kotlinScript { api ->
            api.gradlew("assemble", "--parallel", "--console=plain", "--no-daemon")
        }
    }
}

job("KInference / Test / JVM") {
    container("JVM Tests", jvmContainer) {
        kotlinScript { api ->
            api.gradlew("jvmTest", "--parallel", "--console=plain", "-Pci", "--no-daemon")
        }
    }
}

job("KInference / Test / JS IR") {
    container("JS IR Tests", jsContainer) {
        shellScript {
            content = xvfbRun("./gradlew jsTest --parallel --console=plain -Pci --no-daemon")
        }
    }
}

job("KInference / Test / JS Legacy") {
    container("JS Legacy Tests", jsContainer) {
        shellScript {
            content = xvfbRun("./gradlew jsLegacyTest --parallel --console=plain -Pci --no-daemon")
        }
    }
}

job("KInference / Heavy Test / JVM") {
    container("JVM Heavy Tests", jvmContainer) {
        addAwsKeys()

        kotlinScript { api ->
            api.gradlew("jvmHeavyTest", "--console=plain", "-Pci", "--no-daemon")
        }
    }
}

job("KInference / Heavy Test / JS IR") {
    container("JS IR Heavy Tests", jsContainer) {
        addAwsKeys()

        shellScript {
            content = xvfbRun("./gradlew jsHeavyTest --console=plain -Pci --no-daemon")
        }
    }
}

job("KInference / Heavy Test / JS Legacy") {
    container("JS Legacy Heavy Tests", jsContainer) {
        addAwsKeys()

        shellScript {
            content = xvfbRun("./gradlew jsLegacyHeavyTest --console=plain -Pci --no-daemon")
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
