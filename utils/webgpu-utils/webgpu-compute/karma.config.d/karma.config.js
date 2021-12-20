config.set({
    browsers: ['ChromeWebGPU'],

    customLaunchers: {
        ChromeWebGPU: {
            base: 'Chrome',
            flags: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
        }
    },

    browserDisconnectTimeout: 1200000,
    browserNoActivityTimeout: 1200000,
    pingTimeout: 1200000,
    client: {
        mocha: {
            timeout: 1200000
        }
    }
})
