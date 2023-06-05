from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World! main.py'
@app.route('/api')
def hello_worlds():
    return 'Hello, World! api.py'
@app.route('/api/face')
def hello_worldk():
    return 'Hello, World! face.py'

if __name__ == '__main__':
    app.run()