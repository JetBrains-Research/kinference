job("KInference / Build and Test") {
    host("Build and test") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")

        shellScript("Install Firefox, xvfb and JDK") {
            content = """
                apt-get update && apt-get install firefox xvfb openjdk-11-jdk -y -f
            """.trimIndent()
        }

        kotlinScript("Build with Gradle") { api ->
            api.gradlew("build", "-Pci", "-Pdisable-tests", "--console=plain")
        }

        shellScript("Run tests") {
            content = """
                xvfb-run --auto-servernum ./gradlew -Pci jvmTest jsLegacyTest jsIrTest --console=plain
            """.trimIndent()
        }

        shellScript("Run heavy tests") {
            content = """
                xvfb-run --auto-servernum ./gradlew -Pci jvmHeavyTest jsLegacyHeavyTest jsIrHeavyTest --console=plain
            """.trimIndent()
        }
    }
}

job("KInference / Release") {
    startOn {
        gitPush {
            enabled = false
        }
    }

    container("openjdk:11") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")

        kotlinScript("Publish") { api ->
            api.gradlew("publish")
        }
    }
}
