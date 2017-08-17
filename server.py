#!/usr/bin/python

import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from lib import model_importer
import pandas
import secrets
from urllib.parse import urlparse, parse_qs

PORT_NUMBER = int(os.environ['PORT']) if 'PORT' in os.environ else 8080
AUTO_API_KEY = secrets.token_hex(nbytes=16)
API_KEY = os.environ['API_KEY'] if 'API_KEY' in os.environ else AUTO_API_KEY

model = model_importer.model()


class ServerHandler(BaseHTTPRequestHandler):
    def authorize(self, api_key):
        if api_key != API_KEY:
            self.send_response(401)
            self.end_headers()
            return False
        return True

    def extract_params(self):
        params = parse_qs(urlparse(self.path).query)

        # Requests are allowed to omit a description.
        if ('item[title]' not in params):
            self.send_response(400)
            self.end_headers()
            return False

        title = params['item[title]'][0]
        desc_key = 'item[description]'
        description = params[desc_key][0] if desc_key in params else ''

        return title, description

    def predict(self, title, description):
        X = [{'title': title, 'description': description}]
        df = pandas.DataFrame.from_dict(X)
        y = model.predict(df)
        return y[0]

    def do_POST(self):
        api_key = self.headers['api_key']
        if not self.authorize(api_key):
            return

        if not self.extract_params():
            return

        title, description = self.extract_params()

        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        prediction = self.predict(title, description)
        self.wfile.write(prediction.encode())


try:
    server = HTTPServer(('', PORT_NUMBER), ServerHandler)
    print('Started HTTP server on port' , PORT_NUMBER)

    if API_KEY == AUTO_API_KEY:
        print('Using auto-generated API key:', API_KEY)
        print('Please ensure that client applications provide this key.')
        print('Set the `API_KEY` environment variable to persist a key.')

    server.serve_forever()

except KeyboardInterrupt:
    print('Shutting down web server')
    server.socket.close()
