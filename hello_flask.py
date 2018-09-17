import flask

import vsearch

app = flask.Flask(__name__)


@app.route('/')
def hello() -> str:
    return 'I just want to fuck your juicy pussy!'


@app.route('/search4')
def do_search() -> str:
    return str(vsearch.search4letters('life,the universe,and everything', 'eiru,!'))


app.run()
