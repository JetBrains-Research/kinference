config.middleware = config.middleware || [];
config.middleware.push('resource-loader');

function ResourceLoaderMiddleware() {
    const fs = require('fs');

    return function (request, response, next) {
        const uri = decodeURI(request.originalUrl)
        let path = '';
        if (uri.startsWith('/absolute')) {
            path = uri.slice(9)
        } else if (uri.startsWith('/s3')) {
            path = '../../../../test-data/s3/tests' + uri.slice(3)
        } else {
            path = '../../../../utils/test-utils' + uri
        }
        const content = fs.readFileSync(path);
        response.writeHead(200);
        response.end(content);
    }
}

config.plugins.push({
    'middleware:resource-loader': ['factory', ResourceLoaderMiddleware]
});

config.set({
  browserDisconnectTimeout: 1200000,
  browserNoActivityTimeout: 1200000,
  pingTimeout: 1200000,
  client: {
    mocha: {
      timeout: 1200000
    }
  }
})
